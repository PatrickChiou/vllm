from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from vllm.config import SimulatorConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import (CompletionSequenceGroupOutput, ExecuteModelRequest,
                           Logprob, SequenceData, SequenceGroupMetadata,
                           SequenceOutput)
from vllm.simulator.trace_store import TraceEntry, TraceStore

logger = init_logger(__name__)


@dataclass
class _ReplayPlan:
    """The token stream one request will produce."""
    output_token_ids: Optional[Tuple[int, ...]]
    output_len: int
    filler_token_id: int
    eos_token_id: Optional[int]

    def token_at(self, index: int) -> int:
        if index < self.output_len:
            if self.output_token_ids is not None:
                return self.output_token_ids[index]
            return self.filler_token_id
        # Past the recorded response: end the request. Without an EOS id we
        # keep emitting filler and rely on max_tokens to stop.
        if self.eos_token_id is not None:
            return self.eos_token_id
        return self.filler_token_id


@dataclass
class SimulatorStats:
    num_steps: int = 0
    num_requests: int = 0
    num_trace_hits: int = 0
    num_trace_misses: int = 0
    # Prompt tokens scheduled for prefill, and the subset the model would
    # actually compute after prefix-cache hits.
    num_prefill_tokens: int = 0
    num_computed_prefill_tokens: int = 0
    num_generated_tokens: int = 0
    simulated_time_s: float = 0.0


class Simulator:
    """Produces model outputs from a trace instead of running a model.

    vLLM's scheduler decides, per step, which sequence groups run and whether
    each one samples a token (`do_sample`). The simulator honours that
    contract exactly: every scheduled group gets one
    `CompletionSequenceGroupOutput`, and groups with `do_sample=True` (the
    last prefill chunk and every decode step) get the next trace token.

    The next token is chosen from the number of output tokens the sequence
    already has, so replay stays correct under batching, chunked prefill and
    recompute preemption without keeping a separate cursor.
    """

    def __init__(
        self,
        config: SimulatorConfig,
        trace: TraceStore,
        block_size: int,
        eos_token_id: Optional[int],
        sliding_window: Optional[int] = None,
    ) -> None:
        self.config = config
        self.trace = trace
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.sliding_window = sliding_window
        self.stats = SimulatorStats()
        self._plans: Dict[str, _ReplayPlan] = {}
        self._next_miss_report = 1

    @property
    def num_active_requests(self) -> int:
        """Requests holding replay state (released when the engine reports
        them finished or aborted)."""
        return len(self._plans)

    def step(
        self, execute_model_req: ExecuteModelRequest
    ) -> Tuple[List[SamplerOutput], float]:
        """Returns the outputs of one engine step and its simulated latency
        in seconds. The caller decides how to wait for that latency."""
        for request_id in execute_model_req.finished_requests_ids:
            self._plans.pop(request_id, None)

        outputs: List[CompletionSequenceGroupOutput] = []
        computed_prefill_tokens = 0
        decode_seqs = 0
        for seq_group_metadata in execute_model_req.seq_group_metadata_list:
            assert isinstance(seq_group_metadata, SequenceGroupMetadata), (
                "The simulator expects full SequenceGroupMetadata, not "
                "deltas.")
            if seq_group_metadata.is_prompt:
                computed_prefill_tokens += self._prefill_step(
                    seq_group_metadata)
            else:
                decode_seqs += len(seq_group_metadata.seq_data)

            samples: List[SequenceOutput] = []
            if seq_group_metadata.do_sample:
                for seq_id, seq_data in seq_group_metadata.seq_data.items():
                    token_id = self._next_token(seq_group_metadata.request_id,
                                                seq_data)
                    samples.append(
                        SequenceOutput(parent_seq_id=seq_id,
                                       output_token=token_id,
                                       logprobs={token_id: Logprob(0.0)}))
                    self.stats.num_generated_tokens += 1
            outputs.append(
                CompletionSequenceGroupOutput(samples=samples,
                                              prompt_logprobs=None))

        latency_s = self._latency_s(computed_prefill_tokens, decode_seqs)
        self.stats.num_steps += 1
        self.stats.simulated_time_s += latency_s
        return [SamplerOutput(outputs=outputs)], latency_s

    def _next_token(self, request_id: str, seq_data: SequenceData) -> int:
        plan = self._plans.get(request_id)
        if plan is None:
            plan = self._make_plan(seq_data.get_prompt_token_ids())
            self._plans[request_id] = plan
        return plan.token_at(seq_data.get_output_len())

    def _make_plan(self, prompt_token_ids: Sequence[int]) -> _ReplayPlan:
        self.stats.num_requests += 1
        entry: Optional[TraceEntry] = self.trace.lookup(prompt_token_ids)
        if entry is None:
            self.stats.num_trace_misses += 1
            self._report_misses(len(prompt_token_ids))
            entry = TraceEntry(output_token_ids=None,
                               output_len=self.config.default_output_len)
        else:
            self.stats.num_trace_hits += 1
        return _ReplayPlan(output_token_ids=entry.output_token_ids,
                           output_len=entry.output_len,
                           filler_token_id=self.config.filler_token_id,
                           eos_token_id=self.eos_token_id)

    def _report_misses(self, prompt_len: int) -> None:
        # Report on the 1st, 10th, 100th, ... miss so a mismatched trace is
        # loud without flooding the log.
        misses = self.stats.num_trace_misses
        if misses < self._next_miss_report:
            return
        self._next_miss_report *= 10
        logger.warning(
            "Simulator trace misses: %d of %d requests (latest prompt: %d "
            "tokens). Missed prompts emit %d filler tokens. A miss means the "
            "server tokenized the prompt differently from the trace; build "
            "traces with vllm.simulator.trace_builder.", misses,
            self.stats.num_requests, prompt_len,
            self.config.default_output_len)

    def _prefill_step(self, seq_group_metadata: SequenceGroupMetadata) -> int:
        """Returns how many prompt tokens of this chunk the model computes.

        Mirrors ModelRunner._compute_lens and _compute_for_prefix_cache_hit,
        including recording the prefix-cache hit on the sequence, which is
        what RequestOutput.num_cached_tokens reports.
        """
        seq_data = next(iter(seq_group_metadata.seq_data.values()))
        context_len = seq_data.get_num_computed_tokens()
        seq_len = min(seq_data.get_len(),
                      context_len + (seq_group_metadata.token_chunk_size or 0))
        query_len = seq_len - context_len
        self.stats.num_prefill_tokens += query_len

        computed_block_nums = seq_group_metadata.computed_block_nums
        if computed_block_nums and self.sliding_window is None:
            prefix_cache_len = len(computed_block_nums) * self.block_size
            seq_data.update_num_cached_tokens(prefix_cache_len)
            if context_len < prefix_cache_len < seq_len:
                query_len = seq_len - prefix_cache_len
            elif seq_len <= prefix_cache_len:
                # Full hit: the model runner still computes the last token.
                query_len = 1

        self.stats.num_computed_prefill_tokens += query_len
        return query_len

    def _latency_s(self, prefill_tokens: int, decode_seqs: int) -> float:
        cfg = self.config
        latency_ms = cfg.prefill_ms_per_token * prefill_tokens
        if decode_seqs > 0:
            latency_ms += (cfg.decode_ms_base +
                           cfg.decode_ms_per_seq * decode_seqs)
        return latency_ms / 1000.0
