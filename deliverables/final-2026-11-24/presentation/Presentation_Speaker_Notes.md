# Presentation speaker notes

15 main slides, 9 backups. Main allocation: 810 s (13 min30 s), leaving90 s before15 min. Spoken words including transitions: 1589. Average planned rate: 117.7 words/min. Estimate, not a timed human rehearsal. No presenter split assigned.

## Slide 1 — A Study of Prefix Sharing
in LLM Serving

**Time:** 20 s. **Main message:** Study cache reuse through the serving control path.

We study prefix sharing in LLM serving using a CPU trace simulator integrated with vLLM. Our team is Thomas Lin, Yu-Ting Chung, Patrick Chiou and Yu-Xin Lai at Rice University. We examine homogeneous chat, heterogeneous sources and mixed-workload eviction.

**Transition:** First, why should workload structure affect cache management?

## Slide 2 — Locality makes eviction a workload-dependent decision

**Time:** 50 s. **Main message:** Different return patterns motivate different retention rules.

Prefill constructs attention state for an input prompt. If a later request starts with the same full prefix, it can reuse those KV blocks instead of computing them again. Limited capacity makes retention important: a conversation grows, a common instruction repeats across sessions, and a document may receive a short burst of questions. We ask whether CPU replay can preserve the serving control path, how cache capacity and block granularity affect realized sharing, and whether any baseline or custom policy is robust across workload mixtures. 

**Transition:** The implementation and final evaluation combine several team contributions.

## Slide 3 — Team contributions

**Time:** 40 s. **Main message:** The final artifacts integrate implementation, experiments and validation.

The complete project history now identifies the main component contributions. Thomas worked on mixed workloads, metadata transport and custom policies. Yu-Ting implemented and optimized the simulator and expanded heterogeneous adapters and traces. Patrick contributed the initial Task1 framework and the final AI-assisted report, presentation, testing, bug repair and result integration. Yu-Xin implemented client timing and templates, homogeneous replay and measurements, baseline policies and sweep automation.  

**Transition:** Here is the completed project state supported by the available evidence.

## Slide 4 — Project progress: all required experiment families covered

**Time:** 30 s. **Main message:** Required experiment families are complete; uncertainty remains qualified.

The simulator now passes both frontend modes. Milestone two contains one hundred eight configurations across three arrival seeds. We reuse the validated fifty-one-run heterogeneous study rather than rerunning it unnecessarily. The new mixed evaluation completes both compositions under all eight policies, with one seed per mixture. The applicable simulator suite passes one hundred forty-six tests. 

**Transition:** The architecture explains which real serving components still execute.

## Slide 5 — Trace execution replaces the model, not the serving loop

**Time:** 80 s. **Main message:** The standard control path surrounds a trace-driven executor.

Read the top row from left to right. Preparation normalizes traces and applies the serving template and tokenizer. The client submits causal HTTP requests to the OpenAI-compatible API and engine. The v0 scheduler decides iteration-level work and continuous batches. The lower row shows prefix allocation and eviction, followed by the simulator executor and normal completion processing. The original model executor would load weights and run GPU kernels; our replacement returns recorded token outputs and uses virtual KV geometry. Sequence state, reference counting, prefix lookup and EOS handling remain active. Workload labels are additional metadata, so we do not claim every scheduler line is untouched. No model weights or real KV tensors are needed, but model configuration and tokenizer files are. Synthetic execution delay and real CPU overhead affect timing, so the system is a cache-behavior simulator rather than a calibrated GPU performance model.

**Transition:** An exact-prefix example makes the sharing boundary concrete.

## Slide 6 — Sharing requires the same complete preceding prefix

**Time:** 50 s. **Main message:** Chained identity prevents sharing arbitrary matching chunks.

These two requests begin with the same first two token blocks. Their logical identities include the entire preceding prefix, so those blocks can share KV state. The final blocks differ and must be computed separately. An identical local chunk after a different history would not be the same prefix block. Only full blocks are reusable; partial tails remain outside that matching unit. Under pressure, LRU selects the least recently accessed unused block, LFU the lowest allocator frequency, and FIFO the oldest insertion.  Protected custom-policy tiers also express priority rather than an absolute guarantee of residency.

**Transition:** Preserving these semantics required several specific engineering safeguards.

## Slide 7 — Engineering: problem → safeguard → validation

**Time:** 70 s. **Main message:** The transport fix preserves labels instead of hiding the broken frontend.

Duplicate prompts can have different recorded responses, so exact tokens alone are insufficient; request IDs disambiguate the trace. Recomputing a sequence must not restart its answer, so next-token selection uses the existing output length. Multi-turn replay must wait for the preceding reply, even if the planned arrival is already past. The final regression involved workload metadata: handlers supplied a label that the multiprocessing client did not accept or serialize. We added it through the client, RPC object and engine-side request submission, keeping an optional default for unlabeled requests. Tests verify both HTTP endpoints and both frontend modes, including request and full-block labels. The full simulator suite now passes one hundred forty-six tests. 

**Transition:** The experimental controls determine which comparisons are meaningful.

## Slide 8 — Methodology: common inputs, controlled cache geometry

**Time:** 45 s. **Main message:** The score weights long prompts more than short ones.

We pin the Qwen model configuration and tokenizer. The homogeneous sweep uses matched ShareGPT cohorts and absolute virtual capacities, holding bytes approximately constant when block size changes. Separate sources retain their source-specific contexts and ratio-sized caches. Mixed policies share identical prepared inputs and capacity, start with empty caches, and run in randomized order. Our primary score divides total cached prompt tokens by total prompt tokens; it differs from averaging request fractions. Repeated-seed whiskers are sample standard deviation, while single-run mixed results have no uncertainty estimate. 

**Transition:** First, the matched homogeneous comparison tests whether longer histories actually remain reusable.

## Slide 9 — M2: multi-turn history does not guarantee realized reuse

**Time:** 65 s. **Main message:** Potential history reuse and realized token-weighted sharing are different.

The two modes start from the same two hundred eligible conversations and identical first-turn target arrivals. Single-turn sends only the first prompt and response; multi-turn sends the complete causal conversation. Under LRU and sixteen-token blocks, sharing is 4.41% for single-turn and 3.52% for multi-turn at zero point zero five GiB. At zero point one GiB the values are 4.41% and 4.71%. The bars summarize three arrival seeds. Longer histories create potential overlap, but the cache must retain complete blocks while many requests are active. Token weighting also gives long prompts greater influence than short ones. The request, hit-count and hit-associated gap CDFs in backup slides describe these different populations; none is an unconditional potential-reuse distribution.

**Transition:** The parameter sweep separates capacity from full-block matching granularity.

## Slide 10 — M2: capacity and granularity are separate controls

**Time:** 55 s. **Main message:** Coarser full-block matching can lose short shared prefixes.

The capacity panel keeps sixteen tokens per block, while the granularity panel keeps the virtual byte budget at zero point zero five GiB. Whole-block rounding means capacities are approximately, rather than bit-for-bit, equal across block sizes. Each point is a three-seed mean with sample standard deviation. A larger block requires a longer identical complete prefix and leaves a longer partial tail that cannot be reused. The single-turn eight-token-block cell also shows a policy interaction: FIFO exceeds LRU at the same byte budget. The measured cells show why capacity and block size should not be treated as interchangeable controls.  These findings apply to this cohort, load and model template.

**Transition:** Heterogeneous workloads introduce a different range of return patterns.

## Slide 11 — M3 Task 1: strong effects coexist with near ties

**Time:** 70 s. **Main message:** Policy rankings are conditional on source and tested settings.

The separate-source study has fifty-one configurations and twenty-one thousand eight hundred thirty-four successful requests. The largest highlighted gap is coding-agent replay: LRU reaches forty-eight point five nine percent, compared with thirty-five point four three for LFU. Growing contexts plausibly benefit from retaining recently added prefixes, while frequency can favor older popular blocks, but we have not isolated that mechanism with an ablation. LFU has the highest observed scores in the Bailian chat and coder windows, each with only one repetition. QuALITY's highest mean is FIFO. AgentBank's LRU and LFU differ by only about zero point zero seven percentage points, and Mooncake's by about zero point zero two. Code Alpaca ties. These are conditional observations, not statistically established universal winners.

**Transition:** Custom policies try to combine workload knowledge with observed reuse.

## Slide 12 — Custom policies: labels, own hits or recent outcomes

**Time:** 65 s. **Main message:** Protection is a victim preference, not pinning.

TLRU and TFIFO use static workload tiers, giving agent and Bailian content preference over one-shot labels and protecting prefixes shared across labels. TLRU uses recency with an eight-second hold; TFIFO uses insertion order without expiry. HLRU and HFIFO instead protect a block after its own reuse, without requiring labels. RFIFO observes the last one hundred resolved block outcomes per label. Once that window is full and at least five percent are hits, future blocks can enter its protected FIFO tier. Existing low blocks are not retroactively promoted. In every case, protection changes victim priority rather than absolutely pinning content. If lower tiers are empty, protected unused blocks can still be evicted. 

**Transition:** The two mixtures test whether those choices survive a different token composition.

## Slide 13 — Mixed workload composition changes the observed ranking

**Time:** 80 s. **Main message:** Compare against the strongest baseline in each mixture; one seed limits certainty.

These are new complete raw-validated experiments, rather than copied historical guide values. The skewed cohort is dominated by coding-agent prompt tokens, while the equal-token cohort selects causal prefixes contributing approximately two hundred sixty thousand prompt tokens per label. Equal tokens do not mean equal request counts. The observed results are: RFIFO reaches 60.25 percent on skewed, 11.13 percentage points above FIFO; and RFIFO reaches 41.18 percent on equal-token, 5.33 percentage points above LRU. Each comparison uses the strongest LRU, LFU or FIFO baseline for that mixture. Policy order is randomized and each cache starts empty. Per-label results in the backup material reveal whether aggregate gains conceal harm to a smaller category. One seed per mixture means these rankings are descriptive and their sampling variability is unknown. Static label protection and adaptive admission can respond differently to composition, but neither result establishes a globally optimal policy.

**Transition:** The final handoff makes both verification and reproduction explicit.

## Slide 14 — TA reproduction: a short path before the full sweeps

**Time:** 50 s. **Main message:** The README provides tested commands and external raw-evidence paths.

The README separates a lightweight correctness path from the expensive evaluation. Use Python three point eleven, the documented empty-target editable installation and the pinned model metadata and tokenizer.  The full applicable suite passes one hundred forty-six tests; a small concurrent exact-token engine test provides a smoke check. The experiment driver records controls and resumes only compatible completed attempts, preserving interrupted outputs. Figure generation reads existing artifacts and does not start a server. Complete commands, expected directories and known limitations are in the README; large datasets and caches are not added to Git.

**Transition:** We conclude with what the evidence supports and what remains limited.

## Slide 15 — Three takeaways

**Time:** 40 s. **Main message:** Conditional findings support further controlled evaluation, not a universal winner.

Three conclusions follow. First, CPU trace execution can retain the serving scheduler and cache-management path while bypassing model computation. Second, homogeneous history does not automatically become realized reuse: byte capacity and block granularity matter. Third, heterogeneous locality and mixture composition change the observed value of retention rules. The results remain limited by transformed traces, short production windows and one seed for each mixed cohort. We make no measured GPU speedup or universal-optimality claim. 

**Transition:** Questions; backup slides provide detailed evidence.

## Slide 16 — Backup: M2 request sharing

**Time:** 0 s. **Main message:** LRU · block16 · 0.05 GiB · seed0; per-request population.

Backup evidence; not included in the timed main talk.

**Transition:** Return to questions.

## Slide 17 — Backup: M2 block hits

**Time:** 0 s. **Main message:** Logical full blocks include zero hits; aggregate across residencies.

Backup evidence; not included in the timed main talk.

**Transition:** Return to questions.

## Slide 18 — Backup: M2 realized reuse gaps

**Time:** 0 s. **Main message:** Successful-hit-associated wall time; misses are excluded.

Backup evidence; not included in the timed main talk.

**Transition:** Return to questions.

## Slide 19 — Backup: planned prompt recurrence

**Time:** 0 s. **Main message:** Nominal prompt-only recurrence; no decode blocks or cache/availability model.

Backup evidence; not included in the timed main talk.

**Transition:** Return to questions.

## Slide 20 — Backup: mixed workload composition

**Time:** 0 s. **Main message:** Token balance is different from equal request counts.

Backup evidence; not included in the timed main talk.

**Transition:** Return to questions.

## Slide 21 — Backup: per-workload mixed sharing

**Time:** 0 s. **Main message:** Inspect category regressions as well as the overall score.

Backup evidence; not included in the timed main talk.

**Transition:** Return to questions.

## Slide 22 — Backup: complete Task1 numerical results

**Time:** 0 s. **Main message:** All eight sources are reported without mixing experimental settings.

Backup table; see the report for uncertainty.

**Transition:** Return to questions.

## Slide 23 — Backup: complete mixed results and measured evictions

**Time:** 0 s. **Main message:** Counters are directly measured for the new mixed trials.

Backup table; no mixed variability estimate is available.

**Transition:** Return to questions.

## Slide 24 — Backup: AI assistance and validation scope

**Time:** 0 s. **Main message:** AI assistance is disclosed beyond proofreading.

Backup disclosure; no presenter assignments or human-review claims are inferred.

**Transition:** Return to questions.

