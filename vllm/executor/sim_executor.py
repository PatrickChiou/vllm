import asyncio
import time
from typing import List, Optional, Set, Tuple

from vllm.config import SimulatorConfig
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import ExecuteModelRequest
from vllm.simulator import Simulator, TraceStore
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, GiB_bytes, LayerBlockType,
                        get_dtype_size)

logger = init_logger(__name__)


class SimExecutor(ExecutorBase):
    """Executor that replays a trace instead of running the model.

    Scheduling, block management, prefix caching and output processing run
    unchanged; only the forward pass is replaced by `Simulator.step`. No
    model weights are loaded and no accelerator is used.
    """

    uses_ray: bool = False

    def _init_executor(self) -> None:
        if self.lora_config is not None:
            raise ValueError("The sim executor does not support LoRA.")
        if self.prompt_adapter_config is not None:
            raise ValueError(
                "The sim executor does not support prompt adapters.")
        if self.speculative_config is not None:
            raise ValueError(
                "The sim executor does not support speculative decoding.")
        if self.scheduler_config.is_multi_step:
            raise ValueError("The sim executor does not support multi-step "
                             "scheduling (--num-scheduler-steps > 1).")

        self.sim_config = (self.vllm_config.simulator_config
                           or SimulatorConfig())
        if self.sim_config.trace_path:
            trace = TraceStore.from_file(self.sim_config.trace_path)
        else:
            logger.warning("No --sim-trace-path given; every request will "
                           "emit %d filler tokens.",
                           self.sim_config.default_output_len)
            trace = TraceStore()

        self.simulator = Simulator(config=self.sim_config,
                                   trace=trace,
                                   block_size=self.cache_config.block_size,
                                   eos_token_id=self._get_eos_token_id())

    def _get_eos_token_id(self) -> Optional[int]:
        # Must match the id the engine's StopChecker compares against, which
        # comes from the tokenizer.
        if not self.model_config.skip_tokenizer_init:
            from vllm.transformers_utils.tokenizer import get_tokenizer
            tokenizer = get_tokenizer(
                self.model_config.tokenizer,
                tokenizer_mode=self.model_config.tokenizer_mode,
                trust_remote_code=self.model_config.trust_remote_code,
                revision=self.model_config.tokenizer_revision)
            if tokenizer.eos_token_id is not None:
                return tokenizer.eos_token_id
        eos = getattr(self.model_config.hf_config, "eos_token_id", None)
        if isinstance(eos, (list, tuple)):
            return eos[0] if eos else None
        return eos

    def _kv_cache_block_bytes(self) -> int:
        model_config = self.model_config
        head_size = model_config.get_head_size()
        num_kv_heads = model_config.get_num_kv_heads(self.parallel_config)
        num_layers = model_config.get_num_layers_by_block_type(
            self.parallel_config, LayerBlockType.attention)
        if self.cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype]
        # Keys and values for every attention layer.
        return (2 * num_layers * self.cache_config.block_size * num_kv_heads *
                head_size * get_dtype_size(dtype))

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        block_bytes = max(self._kv_cache_block_bytes(), 1)
        num_gpu_blocks = int(self.sim_config.kv_cache_gb * GiB_bytes //
                             block_bytes)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             block_bytes)
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        # There is no real KV cache to allocate; the block manager tracks
        # virtual blocks only.
        logger.info("Simulated KV cache: # GPU blocks: %d, # CPU blocks: %d",
                    num_gpu_blocks, num_cpu_blocks)

    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> Optional[List[SamplerOutput]]:
        outputs, latency_s = self.simulator.step(execute_model_req)
        if latency_s > 0:
            time.sleep(latency_s)
        return outputs

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise ValueError("The sim executor does not support LoRA.")

    def remove_lora(self, lora_id: int) -> bool:
        raise ValueError("The sim executor does not support LoRA.")

    def pin_lora(self, lora_id: int) -> bool:
        raise ValueError("The sim executor does not support LoRA.")

    def list_loras(self) -> Set[int]:
        return set()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        raise ValueError("The sim executor does not support prompt adapters.")

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise ValueError("The sim executor does not support prompt adapters.")

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise ValueError("The sim executor does not support prompt adapters.")

    def list_prompt_adapters(self) -> Set[int]:
        return set()

    def check_health(self) -> None:
        return

    def shutdown(self) -> None:
        # Called explicitly and again from __del__.
        simulator = getattr(self, "simulator", None)
        if simulator is not None:
            logger.info("Simulator stats: %s", simulator.stats)
            self.simulator = None


class SimExecutorAsync(SimExecutor, ExecutorAsyncBase):

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        outputs, latency_s = self.simulator.step(execute_model_req)
        if latency_s > 0:
            await asyncio.sleep(latency_s)
        return outputs
