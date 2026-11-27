# Instructor Q&A preparation

Answers distinguish current-source tests, historical raw-validated experiments and the new repository-reported rounded sweep. No GPU measurements are claimed. Presenter assignments are not specified.

1. **Why can prefix sharing be studied without GPU inference?**
   Cache allocation and reuse depend on token-prefix identity, request/sequence state and scheduling decisions. Trace tokens replace the model's forward outputs while those control components execute. This characterizes realized reuse; it does not establish calibrated GPU latency or model quality.

2. **Which vLLM components remain active?**
   OpenAI API handling, the v0 engine, iteration-level scheduler, continuous batching, sequence/block management, reference counting, prefix lookup and output completion remain active. Executor selection, workload metadata and several integration paths are modified; we do not claim every line is unchanged.

3. **How are output tokens generated?**
   Exact prompt tokens and request IDs select a recorded response. Each sampled token follows the sequence's current output length, with ordinary EOS/stop processing. Strict trace mode rejects missing matches; fallback fillers are not evidence of a valid exact replay.

4. **How is continuous batching preserved?**
   The ordinary v0 scheduler still forms iteration-level work and sequence metadata. The simulator returns compatible sampler outputs for scheduled sequences; it does not independently batch a precomputed response list outside the engine.

5. **How are duplicate prompts handled?**
   Request-ID-aware lookup distinguishes the same prompt associated with different recorded responses. Preparation and HTTP replay use matching identifiers; tests cover this ambiguity and exact replay behavior.

6. **How is virtual KV capacity calculated?**
   A block's bytes are 2 × attention layers × KV heads/rank × head dimension × tokens/block × dtype bytes. Absolute GiB or an offline working-set ratio determines whole virtual blocks. No physical GPU tensors are allocated; a longest-request feasibility check prevents impossible configurations.

7. **Why use token-weighted sharing?**
   Total cached prompt tokens divided by total prompt tokens measures the proportion of input tokens reused. Mean/median request fractions answer different questions and weight short prompts differently. We report them separately rather than averaging incompatible metrics.

8. **Why can single- and multi-turn outcomes differ unexpectedly?**
   Multi-turn creates potential growing-prefix overlap, but blocks must survive limited capacity, active requests and actual ordering. Token weighting also gives long later prompts more influence. Matched conversations and first-turn targets control the starting cohort; they do not force multi-turn to achieve a higher realized score. A cache-free prompt-only diagnostic finds 57.73% nominal recurring prompt tokens for multi-turn, versus 3.52% realized sharing in the0.05 GiB LRU/block 16 cell. The diagnostic ignores decode blocks/availability and is not a full-serving bound.

9. **What does block size change?**
   Only full identical prefix blocks can be reused. Coarser blocks can lose short matches and leave longer partial tails. We hold bytes approximately constant across 8/16/32-token blocks; allocator overhead and GPU efficiency are not separately measured.

10. **Why might LRU outperform LFU on coding agents?**
    Growing contexts can reuse recently added content soon, while LFU may retain older historically frequent blocks and evict low-count new content. The large measured gap is consistent with this explanation; no ablation proves it as the sole cause.

11. **Why might LFU help Bailian traces?**
    Recurring hot prefixes can benefit from frequency retention despite intervening one-off requests. LFU leads the tested Bailian windows, but each uses one seed/window, synthetic short outputs and anonymous hashes. This does not establish a general production advantage.

12. **How are mixed workloads constructed?**
    Sources must agree on model, tokenizer, revision, template and context. The builder retags IDs, supplies labels, shuffles conversations and assigns new Poisson first-turn times. Later turns remain causal and include think time. Original production timestamps are not retained in the mix.

13. **Why distinguish equal-token from equal-request mixtures?**
    Prompt lengths differ greatly by source. Equal numbers of requests can still let long coding-agent prompts dominate a token-weighted aggregate. Our balanced cohort takes causal prefixes reaching about 260k prompt tokens per label; actual request/token shares and final-turn overshoot are reported.

14. **What do the custom protection rules use?**
    TLRU/TFIFO use static workload/shared-prefix tiers; HLRU/HFIFO use a block's own first reuse; RFIFO uses a configurable per-label outcome window and threshold (defaults100 and5%) admission threshold. Their recency/FIFO order, time expiry and budget rules differ. The detailed source-grounded policy table is supplied.

15. **What happens when protected content fills the cache?**
    Protection orders unused victims rather than pinning them. If lower tiers are empty, a protected block can be evicted. Active reference-counted blocks are excluded; a truly empty free table remains an allocation constraint.

16. **How does RFIFO adapt, and what are its limits?**
    A first hit or unhit eviction resolves a residency outcome. Full recent windows determine admission of new blocks for any qualifying label; existing low blocks are not retroactively promoted. Adaptation has a warm-up delay and depends on labels/outcome sampling; it is not a universal-optimality guarantee.

17. **How are workload labels obtained?**
    Dataset adapters provide them. They pass through API, engine, sequence and block metadata, including the repaired multiprocessing RPC path. We do not implement or evaluate an online workload classifier. Unlabeled requests remain supported with a None default.

18. **Did the multiprocessing regression invalidate the old Task 1 study?**
    The old study explicitly used the direct frontend, which bypassed the missing RPC parameter. Its request-level validation and cohort arithmetic remain valid. The final fix transports labels through the multiprocessing client/RPC/backend rather than discarding them; the applicable suite now passes153 tests, including both frontends.

19. **Can these numbers be generalized to GPU serving?**
    They support conditional conclusions about realized prefix reuse under the tested traces and controls. GPU kernels, numerical model behavior, communication, bandwidth and calibrated timing are absent. Reuse fractions must not be presented as equal GPU speedups or memory reductions.

20. **How can TAs reproduce the project efficiently?**
    Follow the root README: empty-target editable installation, config/tokenizer preparation, a small exact-token smoke check and the simulator test suite. Then use the resumable study driver with fresh output directories or regenerate figures from existing raw results without starting a server. Exact commands, hashes, controls, logs and large-artifact locations accompany the final output.

Additional caution: new mixed cache eviction counts are measured by an opt-in successful-allocation wrapper; historical M2/Task 1 counts remain unavailable. Protection/promotion counters remain unavailable. Null is used instead of zero or a miss-derived estimate. New mixed comparisons have one seed each; no sample SD/CI or reliable small-margin superiority claim is made. The final presentation PDF is exported from the editable PPTX with LibreOffice Impress; the separate shared-source layout companion is also retained.


## Questions about the new commit

21. **Does the five-mixture sweep overturn the earlier RFIFO result?**
    It broadens the workloads/settings and shows that RFIFO is not always highest. It does not rerun identical historical cohorts. HLRU h30/b1 leads the reported macro mean, while RFIFO w100/t.02 has the smallest worst regret; both are in-sample objectives.

22. **Were all 125 new cells independently validated here?**
    No. The committed guide reports completion, zero failures and no clamping. Only rounded tables and plots are available here. The 153-test suite verifies source behavior, not completion of those experiments. Request logs, complete cohort IDs and source fingerprints are needed to certify the grid.

23. **Can TAs resume the grid safely?**
    Use a fresh output directory and verify expected IDs/counts. The driver’s resume predicate accepts a nonempty successful prefix, so it alone cannot distinguish a complete run from an interrupted prefix.

24. **What changed in reply preparation?**
    Configured EOS/stop token IDs are removed from recorded reply tokens, preventing premature termination relative to expected trace length. Existing prepared traces remain unchanged; future preparations can change outputs, prefix identities and capacity.

25. **Are macro means token weighted across mixtures?**
    No. Each of five mixture scores receives equal weight. Regret compares with the best tested setting in each mixture. Scores are rounded, single-seed and selected in-sample; neither significance nor held-out robustness follows.
