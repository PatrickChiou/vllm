# Significant AI prompts

This archive contains the two actual English user-provided instruction documents for the final implementation/experiment/report/presentation phase. It does not reconstruct unavailable historical prompts from other contributors or assert completed human review. System/developer instructions and account authentication information are excluded. The paper appendix summarizes the scope and quotes selected significant instructions.

## User document 1 — verbatim

```text
# COMP536 Final Project — Complete Implementation, Experiments, and MLSys 2025 Technical Report

## 0. Project Information and Objective

Act as a senior ML systems research engineer and technical paper editor.

Complete the final implementation validation, missing experiments, research analysis, technical report, and presentation for our Rice University COMP536 course project.

**Project Title:** A Study of Prefix Sharing in LLM Serving

**Affiliation:** Rice University

**Course:** COMP536

**Submission Deadline:** November 30, 2026, 11:59 PM, subject to the official Canvas deadline.

### Confirmed Authors

Use the following exact author names, email addresses, and author order in the final technical report.

1. Thomas Lin — tl183@rice.edu
2. Yu-Ting Chung — yc223@rice.edu
3. Patrick Chiou — pc82@rice.edu
4. Yu-Xin Lai — yl371@rice.edu

Git author mapping:

- Thomas Lin → Thomas Lin
- Yu-Ting Chung → Yu-Ting Chung
- PatrickChiou → Patrick Chiou
- cryu1211 → Yu-Xin Lai

All authors are affiliated with Rice University.

Do not reorder authors based on commit counts.

Do not designate a corresponding author without confirmation.

### Team Contributions

Determine each member's technical contributions from the complete Git history, implementation changes, experiment scripts, documentation, and available project artifacts.

For Patrick Chiou, additionally include:

- Final technical report preparation and organization.
- Final presentation preparation and organization.
- Final implementation testing and validation.
- Final bug investigation and fixes.
- Integration and consistency checking of final experimental results.

Distinguish confirmed responsibilities from completed work. Do not claim that a planned fix or experiment has already been completed.

Do not estimate contribution percentages based on commit counts.

If earlier history is unavailable, explicitly identify the limitation instead of inventing contributions.

### Current Project Status

The following items require investigation:

- Latest authoritative Git branch: unknown.
- Milestone 2 experimental completion: unknown.
- Milestone 3 Task 2 raw experimental results: unknown.
- Multiprocessing regression status: unknown.

Do not assume that previously reported problems remain unresolved.

## 1. Inspect the Repository and Project Requirements

Before modifying any implementation or running experiments, inspect:

- `project(1).pdf`: official project requirements.
- `Prefix_Sharing_Technical_Report.pdf`: original technical report.
- `Prefix_Sharing_MLSys2025.pdf`: revised report, if available.
- `Prefix_Sharing_LaTeX.zip`: revised LaTeX source, if available.
- `Prefix_Sharing_Final_Presentation.pptx`: existing presentation.
- `mlsys2025style.zip`: official LaTeX template.
- README and milestone documentation.
- All implementation files.
- Unit tests and integration tests.
- Experiment scripts.
- Raw and aggregated experimental results.
- Git branches, remotes, tags, and complete commit history.

Locate equivalent files if their names or paths differ.

Inspect the Git repository using appropriate commands, including:

- `git status`
- `git branch -a`
- `git remote -v`
- `git log --all`
- `git shortlog -sne --all`
- `git rev-parse --is-shallow-repository`

Determine whether the repository contains complete history.

Inspect remote branch information where access is available. Fetch missing history if appropriate and authorized.

Do not blindly switch branches, reset files, overwrite uncommitted changes, or force-push.

If multiple branches contain conflicting implementations, identify the differences and determine which branch should serve as the final integration branch.

Preserve all original Git history.

The course requires submission of the code with its `.git` history and a README containing setup and execution instructions.

### Requirements-to-Evidence Checklist

Create a checklist covering:

- Milestone 1: CPU-based vLLM simulator.
- Milestone 2 Task 1: Client simulator.
- Milestone 2 Task 2: Single-turn and multi-turn ShareGPT experiments.
- Milestone 2 Task 3: Cache parameter sensitivity.
- Milestone 3 Task 1: Separate heterogeneous workload experiments.
- Milestone 3 Task 2: Mixed-workload custom eviction policies.
- Implementation correctness tests.
- Reproducibility.
- Technical report.
- AI Assistance disclosure.
- Final presentation.

Classify each requirement as:

- Completed and verified.
- Implemented but not experimentally verified.
- Partially completed.
- Missing.
- Blocked.

Search for existing experimental artifacts before scheduling any new experiments.

Do not rerun valid experiments unnecessarily.

## 2. Milestone 1 — Simulator Implementation and Validation

Inspect the actual CPU-based simulator implementation integrated into the vLLM v0 online serving architecture.

The course specifies the original vLLM commit:

`94d545a1a18e20ea8763a6760194589b8a3c9065`

Verify the repository's relationship to this upstream commit.

Do not reset the modified project to the upstream commit.

### 2.1 Verify the Simulator Architecture

Confirm that the implementation preserves the intended serving control path:

- Online OpenAI-compatible API.
- Request management.
- AsyncLLMEngine.
- Iteration-level scheduling.
- Continuous batching.
- Sequence management.
- Prefix cache lookup.
- Virtual KV cache block allocation.
- Reference counting.
- Cache eviction.
- Output processing.

Actual GPU inference should be replaced by trace-driven token replay.

Verify that the simulator does not require model weights or actual GPU KV tensors for the studied configurations.

Document any remaining dependencies on model configuration and tokenizer files.

### 2.2 Investigate Multiprocessing Regression

The previous audit reported that serving handlers passed a `workload` argument while `MQLLMEngineClient.generate()` did not accept it.

The historical test result was:

- 136 passed.
- 5 failed.

First determine whether this problem has already been fixed in the latest repository.

If unresolved, trace workload metadata through:

1. OpenAI API handlers.
2. AsyncLLMEngine.
3. MQLLMEngineClient.
4. Multiprocessing serialization and transport.
5. Engine-side request handling.
6. Sequence metadata.
7. Prefix cache block metadata.
8. Workload-aware eviction policies.

Implement the minimal correct fix.

Do not simply discard the workload argument or globally disable multiprocessing to hide the problem.

Maintain backward compatibility for requests without workload labels.

Validate both direct and multiprocessing frontends.

### 2.3 Correctness Testing

Run relevant unit tests and integration tests, followed by the full applicable test suite.

Verify:

- Exact prompt and request-ID trace matching.
- Duplicate prompts with different recorded responses.
- Prefill and decode behavior.
- Continuous batching.
- Scheduler behavior.
- EOS and stop handling.
- Preemption and recomputation.
- Prefix cache hits.
- Virtual KV cache capacity.
- FIFO and LFU metadata persistence.
- Workload metadata propagation.
- Custom eviction policy behavior.
- Request completion and cleanup.

Investigate the previously reported Ruff B905 warning if it remains relevant.

Record actual test results.

Do not claim that all tests pass unless they actually pass.

Save the test commands, logs, and validation summary.

## 3. Milestone 2 — Homogeneous Chatbot Workload

This milestone is required by the instructor.

The previous report indicated that preprocessing, replay, metrics, and parameter-sweep implementations existed, but complete experimental outputs were unavailable.

First recover and validate existing results.

If the required results are missing, execute the following experiments.

### 3.1 Experiment M2-A: Single-Turn vs. Multi-Turn

Use the ShareGPT dataset and the existing client simulator.

Single-turn mode:

- Replay only the first user prompt and corresponding assistant response of each eligible conversation.

Multi-turn mode:

- Replay all eligible turns.
- Preserve causal conversation dependencies.
- Submit subsequent turns only after the preceding response completes.

Use matched eligible conversation cohorts.

Control:

- Model and revision.
- Tokenizer.
- Chat template.
- Context length.
- Block size.
- Absolute virtual cache capacity.
- Eviction policy.
- Arrival-generation methodology.
- Concurrency.
- Random seeds.

Use the original documented default configuration where available.

Use at least three arrival seeds where practical.

If the original configuration cannot be recovered, explicitly document the replacement configuration.

Collect:

1. Per-request cached prompt token fraction.
2. Token-weighted aggregate prefix sharing rate.
3. Mean and median per-request sharing fraction.
4. Hits per logical full prefix block, including zero-hit blocks.
5. Hit-associated reuse time intervals.
6. Total prompt tokens.
7. Total cached prompt tokens.
8. Successful and failed request counts.
9. Eviction counts, if available.
10. Actual dispatch rate and dispatch lag.
11. Virtual cache capacity in blocks and bytes.

Generate:

- CDF of per-request prefix sharing fraction.
- CDF of hits per logical cache block.
- CDF of reuse time intervals.
- Aggregate sharing-rate comparison between single-turn and multi-turn modes.

Clearly distinguish realized cache-hit intervals from unconditional potential reuse intervals.

If practical, calculate policy-independent potential reuse intervals from the traces and present them separately.

Analyze why single-turn and multi-turn workloads exhibit different reuse patterns.

### 3.2 Experiment M2-B: Cache Parameter Sensitivity

Inspect the existing parameter-sweep implementation.

The previous report described a 36-configuration sweep.

Recover and reuse its original configuration if valid.

If the original sweep configuration is unavailable, use this fallback:

- Conversation mode: single-turn and multi-turn.
- Cache capacity ratio: 0.10, 0.25, 0.50.
- Block size: 16 and 32 tokens.
- Eviction policy: LRU, LFU, FIFO.

This produces 36 configurations per arrival seed.

Complete at least one seed for all configurations before expanding repetitions.

Use three seeds where runtime permits.

When comparing block sizes, hold virtual cache capacity in bytes approximately constant rather than holding the number of blocks constant.

When comparing cache capacity ratios, document:

- Working-set definition.
- Actual allocated blocks.
- Actual virtual capacity.
- Longest-request feasibility clamp.

Do not silently compare configurations with incompatible capacity definitions.

Reuse overlapping M2-A and M2-B configurations where their experimental settings are identical.

Generate:

- Prefix sharing rate versus cache capacity.
- Prefix sharing rate versus block size.
- Prefix sharing rate by eviction policy.
- Separate single-turn and multi-turn comparisons.
- Mean and sample standard deviation across available seeds.

Analyze:

- Cache capacity saturation.
- Block granularity.
- Partial-block reuse.
- Recency versus frequency.
- Interactions between workload structure and eviction policy.

Do not assume monotonic trends or invent missing results.

## 4. Milestone 3 Task 1 — Separate Heterogeneous Workloads

The previous verified study reported:

- Eight workload sources.
- 51 experimental configurations.
- 21,834 successful replayed requests.
- Zero replay validation failures.
- Three baseline policies: LRU, LFU, FIFO.
- Block size: 16 tokens.
- Cache ratio: approximately 0.25.
- Concurrency: 8.

The workload sources were:

1. ShareGPT.
2. AgentBank.
3. CC-Bench coding agent.
4. QuALITY document QA.
5. Bailian Chat.
6. Bailian Coder.
7. Mooncake Tool Agent.
8. Code Alpaca.

### 4.1 Validate Existing Results

Do not automatically rerun the entire study.

Locate the original raw logs and validation artifacts.

Check:

- Workload hashes.
- Trace hashes.
- Model revision.
- Prompt-token totals.
- Cache capacity.
- Arrival seeds.
- Request-level validation.
- Aggregate metric calculations.

Confirm whether the latest implementation changes affect the validity of existing experiments.

Run targeted regression experiments if necessary.

Historical reference results include:

| Workload | LRU | LFU | FIFO |
|---|---:|---:|---:|
| ShareGPT | 18.81% | 18.92% | 18.21% |
| AgentBank | 50.55% | 50.47% | 49.44% |
| Coding Agent | 48.59% | 35.43% | 47.35% |
| QuALITY | 49.25% | 45.54% | 50.35% |
| Bailian Chat | 28.44% | 29.52% | 26.75% |
| Bailian Coder | 35.87% | 37.28% | 32.88% |
| Mooncake | 40.27% | 40.25% | Retrieve from raw logs |
| Code Alpaca | 29.53% | 29.53% | 29.53% |

These are historical reference values, not targets that new experiments must reproduce exactly.

Do not adjust experimental settings to artificially reproduce these numbers.

### 4.2 Improve Task 1 Analysis

Generate a complete numerical table directly from validated raw results.

Include:

- Mean token-weighted sharing rate.
- Sample standard deviation.
- Number of repetitions.
- Highest observed policy.
- Difference between the best and second-best policies.

Generate publication-quality figures:

- Baseline comparison across eight workloads.
- Per-request sharing fraction CDF.
- Block hit-count CDF.
- Reuse interval CDF.

Explain the observed differences using workload characteristics:

- Growing conversation histories.
- Recency versus historical access frequency.
- Repeated system prompts.
- Shared document prefixes.
- Short-lived and long-lived reuse patterns.

In particular, investigate why LRU substantially outperforms LFU on coding-agent workloads.

Distinguish measured findings from proposed mechanisms.

Do not describe near ties as statistically significant improvements.

Do not claim a universally optimal eviction policy.

## 5. Milestone 3 Task 2 — Mixed Workloads

The previous report contained repository-reported mixed-workload results, but complete raw logs were unavailable.

Determine whether these results have since been recovered or reproduced.

If valid raw logs exist, reuse them.

Otherwise, rerun the missing experiments.

### 5.1 Inspect Custom Eviction Policies

Evaluate the actual implementations of:

- LRU.
- LFU.
- FIFO.
- TLRU.
- TFIFO.
- HLRU.
- HFIFO.
- RFIFO.

Document each policy's:

- Eviction priority.
- Workload label usage.
- Protection criteria.
- Protection duration.
- Hit-history tracking.
- Shared-block handling.
- Protected-block budget.
- Metadata overhead.
- Computational complexity.
- Behavior when no unprotected blocks remain.

Verify the actual algorithms before writing pseudocode.

Do not infer implementation details solely from policy names.

Explain how workload labels are assigned.

Clearly distinguish dataset-provided labels from online workload classification.

### 5.2 Experiment M3-B1: Skewed Mixed Workload

Recover the original skewed workload construction procedure.

The historical mixture reportedly contained approximately 79.2% coding-agent prompt tokens.

Historical reference results:

- LRU: 46.1%.
- LFU: 20.4%.
- FIFO: 51.7%.
- TLRU: 43.5%.
- TFIFO: 60.4%.
- HLRU: 42.6%.
- HFIFO: 37.8%.
- RFIFO: 58.3%.

Record the actual request-level and token-level workload composition.

Do not artificially tune the mixture to reproduce historical results.

### 5.3 Experiment M3-B2: Equal-Token Mixed Workload

Recover or construct a mixture with approximately balanced prompt-token contributions across workload categories.

Equal-token composition must not be confused with equal request counts.

Historical reference results:

- LRU: 37.8%.
- LFU: 33.3%.
- FIFO: 36.8%.
- TLRU: 38.2%.
- TFIFO: 29.5%.
- HLRU: 38.0%.
- HFIFO: 32.9%.
- RFIFO: 38.4%.

Use the original documented mixed-workload configuration where available:

- Block size: 16 tokens.
- Cache ratio: 0.10.
- Context length: 32768.
- Target request rate: 100.
- Concurrency: 8.

Use identical prepared workloads, traces, and cache capacities across policies within each mixture and seed.

Randomize policy execution order for new experiments.

Start each policy run with a fresh server and empty cache.

Complete one seed for both mixtures and all eight policies before expanding to three seeds.

Three seeds produce 48 total runs across both mixtures.

### 5.4 Mixed-Workload Metrics

Collect:

- Overall token-weighted sharing rate.
- Per-workload token-weighted sharing rate.
- Per-request sharing fraction.
- Total cached prompt tokens.
- Total prompt tokens.
- Request-level workload composition.
- Token-level workload composition.
- Eviction counts.
- Protection and promotion statistics, if available.
- Dispatch lag.
- Achieved request rate.
- Request validation failures.

Generate:

- Grouped bar chart comparing all eight policies under both mixtures.
- Per-workload sharing-rate comparison.
- Workload composition chart.
- Mean and sample standard deviation.
- Comparison against the strongest baseline in each mixture.

Report percentage-point differences and relative improvements separately.

Investigate whether overall improvements conceal performance degradation for individual workload categories.

Analyze why static workload-aware policies may perform differently under skewed and balanced mixtures.

Evaluate whether adaptive protection improves robustness.

Do not claim that RFIFO or any other policy is universally superior without sufficient evidence.

### 5.5 Optional Robustness and Ablation Experiments

Only after completing required experiments, evaluate additional configurations if time permits.

Suggested priorities:

1. RFIFO recent-hit window: 50, 100, 200.
2. RFIFO hit-rate threshold: 2%, 5%, 10%.
3. TLRU/HLRU protection duration: 2, 4, 8, 16 seconds.
4. Mixed cache ratio: 0.10, 0.25, 0.50.
5. Intermediate coding-agent token contributions.

Inspect the actual supported parameters before executing these experiments.

Use controlled one-factor-at-a-time comparisons rather than a large Cartesian product.

Separate policy tuning from held-out evaluation where practical.

Do not prioritize optional ablations over required milestones.

## 6. Experimental Integrity and Reproducibility

For every new experiment, preserve:

- Exact execution command.
- Git commit and dirty-working-tree status.
- Model and tokenizer revision.
- Dataset identifier and revision.
- Prepared workload manifest.
- Trace hash.
- Workload hash.
- Random seed.
- Block size.
- Virtual cache capacity.
- Eviction policy.
- Arrival configuration.
- Concurrency.
- Raw request-level results.
- Raw block-level results where available.
- Server logs.
- Aggregated metrics.
- Validation status.

Store new experiments in separate output directories.

Never overwrite historical results.

Make interrupted experiment batches resumable.

Calculate the primary metric as:

Token-weighted prefix sharing rate = total cached prompt tokens / total prompt tokens.

Distinguish this metric from the arithmetic mean of per-request sharing fractions.

Validate:

- Cached prompt tokens never exceed prompt tokens.
- Metrics are finite and bounded.
- Request IDs are unique within each run.
- Compared policies receive identical prepared inputs.
- Cache capacities are comparable.
- Failed requests are explicitly reported.
- Aggregated metrics match raw logs.
- CDFs include the correct populations.
- Figures and tables are generated from validated results.

If failures differ between policies, do not silently compare successful-only cohorts without explaining the resulting bias.

Report CPU simulation measurements as cache behavior.

Do not claim measured GPU latency, GPU throughput, or actual GPU memory improvements unless independently measured.

## 7. Technical Report Revision

Use the official MLSys 2025 LaTeX template supplied by the team.

Modify the existing LaTeX source rather than editing the PDF directly.

The main report must not exceed six pages, excluding references and appendices.

Use the exact confirmed author names, emails, and author order provided in Section 0.

Use Rice University as the affiliation.

Use a formal systems research paper style.

The paper should focus on technical contributions, research methodology, experimental findings, and implications.

Avoid repeatedly using repository-audit language in the main research narrative.

Do not hide genuine limitations or missing evidence.

### Recommended Paper Structure

**Abstract**

Summarize:

- Research motivation.
- CPU-based trace-driven simulator.
- Experimental methodology.
- Homogeneous workload findings.
- Heterogeneous workload findings.
- Mixed-workload policy findings.
- Main research implications.

Use only validated results.

**1. Introduction**

Explain:

- Why prefix sharing matters.
- Why cache eviction affects reuse.
- Why workload characteristics matter.
- Research questions.
- Main technical and empirical contributions.

Emphasize the following contributions where supported:

1. CPU-based online serving simulator.
2. Reproducible heterogeneous workload evaluation framework.
3. Characterization of prefix reuse and cache eviction behavior.
4. Workload-aware and reuse-aware eviction policy design.

**2. Background and Related Work**

Explain:

- KV cache.
- Prefill and decode.
- Prefix sharing.
- PagedAttention.
- Block-based cache management.
- LRU, LFU, and FIFO.

Consider relevant research on vLLM, SGLang, Prompt Cache, CacheBlend, and Mooncake.

Verify every academic reference before citing it.

Do not fabricate citations or exaggerate novelty.

**3. System Design and Implementation**

Explain:

- Original vLLM execution path.
- CPU simulator architecture.
- SimExecutor and trace replay.
- Scheduler and continuous batching.
- Virtual KV cache management.
- Prefix block identity.
- Request-ID trace matching.
- Causal client replay.
- Workload metadata propagation.
- Custom eviction policy algorithms.

Include:

- A clear architecture diagram comparing original vLLM with the proposed simulator.
- A prefix-sharing illustration.
- Concise pseudocode or a comparison table for custom eviction policies.
- Relevant equations and metric definitions.

**4. Experimental Methodology**

Include:

- Dataset characteristics.
- Model configuration.
- Cache configuration.
- Experimental controls.
- Arrival methodology.
- Random seeds.
- Evaluation metrics.
- Validation procedures.

Use concise configuration tables.

**5. Experimental Results**

Organize into:

5.1 Homogeneous Chatbot Workload.

5.2 Cache Parameter Sensitivity.

5.3 Heterogeneous Workload Eviction Policies.

5.4 Mixed-Workload Eviction Policies.

5.5 Policy Robustness and Ablation, if completed.

For each subsection, explain:

- Research question.
- Experimental setup.
- Main observations.
- Possible mechanisms.
- Relevant limitations.

Prioritize figures that directly answer the instructor's required research questions.

Include the most important figures in the main paper.

Move supplementary plots and detailed tables to the appendix.

**6. Discussion and Limitations**

Discuss:

- Workload-dependent temporal locality.
- Recency versus frequency.
- Static versus adaptive eviction.
- Sensitivity to workload composition.
- Trace transformations.
- Limited repetitions.
- CPU simulation versus actual GPU performance.
- Remaining implementation or experimental limitations.

Keep this section concise and accurate.

Remove outdated limitations only when new evidence demonstrates that they have been resolved.

**7. AI Assistance**

Use concise, professional wording focused on verification, experimental validation, data organization, and report preparation.

Preferred wording:

AI tools were used to assist with implementation verification, experimental result validation, data organization, and technical report preparation. Specifically, AI helped review implementation consistency, identify potential issues, organize experimental findings, and improve the clarity of technical explanations and visualizations.

If AI was also used to generate or modify implementation code, tests, experiment scripts, or figures, accurately disclose that assistance.

Do not falsely imply that AI was used only for proofreading if it also contributed to implementation.

Include significant AI prompts in the appendix, as required by the course instructions.

Do not claim that team members reviewed or fully understood generated code unless confirmed.

**8. Conclusion**

Answer the main research questions using validated findings.

Emphasize:

- The simulator's technical contribution.
- The impact of cache configuration.
- Workload-dependent eviction behavior.
- The effectiveness and limitations of custom policies.
- Implications for future workload-aware LLM serving systems.

**References**

Include verified academic papers, datasets, and relevant systems documentation.

**Appendix**

Include:

- Supplementary figures.
- Full numerical results.
- Detailed experiment configurations.
- Reproduction commands.
- Significant AI prompts.
- Supplementary validation details.
- Team contributions.

## 8. Team Contributions Section

Use the confirmed author mapping and complete Git history.

Inspect actual diffs and implementation ownership rather than relying only on commit messages.

The following are preliminary leads from the previous audit, not definitive contribution statements:

**Thomas Lin**

Investigate contributions related to mixed-workload construction, workload metadata propagation, custom eviction policies, and mixed-workload evaluation.

**Yu-Ting Chung**

Investigate contributions related to heterogeneous dataset adapters, workload preparation, production traces, and expanded Task 1 experiments.

**Patrick Chiou**

Investigate contributions related to initial Task 1 experiments, baseline evaluation, testing, documentation, and implementation.

Additionally include confirmed responsibilities for final report and presentation preparation, final testing, validation, and bug fixes.

**Yu-Xin Lai**

Inspect the full Git history associated with `cryu1211`, including earlier Milestone 1 and Milestone 2 implementation, testing, experiments, and documentation where supported.

Do not assign Milestone 1 or Milestone 2 ownership to Yu-Xin Lai unless supported by evidence.

Write a concise and balanced contribution statement for each author.

Avoid unsupported claims and numerical effort percentages.

## 9. Update the Final Presentation

After validating experimental results and updating the technical report, revise the existing PowerPoint presentation.

Use the confirmed author names.

Maintain consistency between the presentation and technical report.

Include:

1. Research motivation.
2. Research questions.
3. Simulator architecture.
4. Prefix-sharing mechanism.
5. Experimental methodology.
6. Milestone 2 single-turn versus multi-turn results.
7. Cache parameter sensitivity.
8. Milestone 3 Task 1 baseline comparison.
9. Workload locality analysis.
10. Custom eviction policies.
11. Mixed-workload results.
12. Discussion and limitations.
13. Team contributions.
14. Conclusion.

Adjust slide count and speaker notes to fit the existing approximately 15-minute presentation target.

Do not invent presenter assignments.

Use clear figures, readable labels, concise slide text, and natural English speaker notes.

Remove outdated audit statements when new validated evidence resolves the corresponding issues.

Do not remove unresolved limitations.

## 10. Final Compilation and Quality Assurance

Compile the technical report using the official MLSys 2025 style.

Verify:

- Main text is no longer than six pages.
- References and appendix are excluded from the main-text page count.
- All four author names and emails are correct.
- Author order is correct.
- Rice University affiliation is correct.
- No unresolved citations.
- No missing figures.
- No broken cross-references.
- No overlapping text or tables.
- No clipped figures.
- Figures are readable at normal PDF zoom.
- Numerical values match validated experimental outputs.
- Abstract and conclusion agree with the actual results.
- AI Assistance is a dedicated section.
- Significant AI prompts are included in the appendix.

Render and visually inspect every PDF page.

Do not merely report successful LaTeX compilation without inspecting the final layout.

Also verify the presentation layout and consistency with the paper.

## 11. Required Deliverables

Produce or update:

1. `Prefix_Sharing_Final_Report.pdf`
2. Complete MLSys 2025 LaTeX source.
3. Bibliography.
4. Publication-quality figures.
5. `Prefix_Sharing_Final_Presentation.pptx`
6. Raw experimental outputs.
7. Aggregated CSV or JSON results.
8. Experiment reproduction scripts.
9. Updated README.
10. Test validation summary.
11. Requirements completion checklist.
12. Final changelog.
13. Team contribution summary.

Preserve the original reports, experimental artifacts, and Git history.

Use a clearly organized output directory.

Do not commit unnecessary model weights, caches, or large generated datasets.

Do not push changes to a remote repository without explicit authorization.

## 12. Execution Priorities and Runtime Management

### P0 — Required

1. Inspect latest repository state.
2. Recover existing experimental artifacts.
3. Confirm complete Git history and team contributions.
4. Investigate and fix unresolved implementation bugs.
5. Validate the simulator.
6. Complete missing Milestone 2 experiments.
7. Recover or reproduce required mixed-workload experiments.
8. Validate existing Task 1 results.
9. Update figures and tables.
10. Rewrite and compile the final report.
11. Update the final presentation.

### P1 — Important

- Repeat experiments across multiple seeds.
- Add per-workload mixed results.
- Improve algorithm explanations.
- Improve research discussion.
- Improve reproducibility.

### P2 — Optional

- Additional ablations.
- Additional workload compositions.
- Additional cache capacity levels.
- Additional robustness experiments.

Before starting expensive experiments:

- Estimate runtime using a small pilot.
- Check available CPU, memory, disk space, and dependencies.
- Prioritize required experiments.
- Reuse valid existing results.
- Make experiment batches resumable.

Do not reduce dataset size or alter experimental configurations merely to obtain faster results without documenting the change.

If resource constraints prevent the full evaluation, complete the highest-priority valid configurations and clearly report what remains missing.

Do not stop after producing an audit or proposed plan.

Proceed with implementation, testing, missing experiments, analysis, and report generation wherever the environment permits.

If blocked, identify the exact blocker and provide the commands needed to resume.

## 13. Final Completion Summary

At completion, provide a concise summary containing:

- Final branch and Git commit.
- Implementation changes.
- Bugs fixed.
- Remaining bugs.
- Tests passed and failed.
- Existing experiments reused.
- New experiments executed.
- Experimental runtime.
- Newly validated results.
- Main research findings.
- Remaining incomplete requirements.
- Final report page count.
- Final presentation slide count.
- Final artifact locations.
- Reproduction commands.

Clearly distinguish completed work from planned work.

Do not fabricate results, contributions, citations, or validation evidence.

The final objective is a technically correct, reproducible, and academically credible COMP536 project submission.
```

## User document 2 — verbatim

```text
## 9. Final Presentation — Required Deliverable

Update the existing PowerPoint presentation according to the instructor's official Final Presentation requirements.

The presentation is a required deliverable, not an optional task.

### 9.1 Official Instructor Requirements

The instructor specifies:

- Each team must present for no more than 15 minutes.
- The presentation may be delivered by one or multiple team members.
- Presentation quality accounts for 5 out of 40 project points.
- TAs will run the submitted code, so clear execution instructions are required.
- The presentation should serve as a checkpoint of the project's progress.

The presentation must explicitly answer:

1. How is labor divided among team members?
2. What is the current project progress?
3. What does the current system look like?
4. What are the main system components and workflow?
5. What technical challenges were encountered?
6. How were those challenges addressed?
7. What research findings have been obtained?
8. How can the TAs run and reproduce the project?

Use figures and diagrams to communicate research findings wherever possible.

### 9.2 Confirmed Team Information

Project title:

A Study of Prefix Sharing in LLM Serving

Rice University — COMP536

Use the following exact author names and order:

- Thomas Lin — tl183@rice.edu
- Yu-Ting Chung — yc223@rice.edu
- Patrick Chiou — pc82@rice.edu
- Yu-Xin Lai — yl371@rice.edu

Determine technical contributions from the complete Git history and available project artifacts.

Additionally include Patrick Chiou's responsibilities for final report preparation, presentation preparation, implementation testing, validation, and final bug fixes.

Distinguish completed contributions from assigned responsibilities.

Do not invent presenter assignments or contribution percentages.

### 9.3 Presentation Structure and Timing

Target a total presentation duration of approximately 13 minutes and 30 seconds, leaving a 90-second buffer before the instructor's 15-minute limit.

Use approximately 15 main slides.

Additional backup slides may be included after the conclusion for questions.

**Slide 1 — Title (20 seconds)**

- Project title.
- Four team members.
- Rice University.
- COMP536.

**Slide 2 — Motivation and Research Questions (50 seconds)**

Explain:

- Why redundant prefill computation matters.
- How prefix sharing improves KV cache reuse.
- Why different workloads may require different eviction policies.
- The project's main research questions.

**Slide 3 — Team Contributions (40 seconds)**

Present the four team members and their verified technical contributions.

Use a concise table.

Base contributions on actual Git history and confirmed responsibilities.

**Slide 4 — Project Progress (30 seconds)**

Summarize:

- Milestone 1: CPU-based simulator.
- Milestone 2: Homogeneous chatbot workload.
- Milestone 3 Task 1: Heterogeneous workloads.
- Milestone 3 Task 2: Mixed-workload eviction policies.

Clearly distinguish completed, validated, partially completed, and remaining work.

Update this slide after completing the final experiments.

**Slide 5 — System Architecture (80 seconds)**

Create a clear workflow diagram showing:

- Dataset and trace preprocessing.
- Client replay.
- OpenAI-compatible API.
- vLLM v0 engine.
- Scheduler and continuous batching.
- Virtual KV cache manager.
- Eviction policy.
- Trace-driven simulator.
- Metrics collection.

Highlight which components were preserved from vLLM and which were modified or replaced.

Explain how GPU inference is bypassed while preserving the serving control path.

**Slide 6 — Prefix Sharing and Cache Management (50 seconds)**

Use a visual example showing:

- Two requests with identical prefixes.
- Shared KV cache blocks.
- Cache hits and misses.
- Chained block identity.
- Cache eviction under limited capacity.

Briefly explain LRU, LFU, and FIFO.

**Slide 7 — Engineering Challenges and Solutions (70 seconds)**

Present the most important verified technical challenges.

Potential topics include:

- Replacing GPU execution without breaking the scheduler.
- Matching recorded responses to requests.
- Duplicate prompts with different responses.
- Preserving causal multi-turn replay.
- Maintaining cache metadata.
- Supporting heterogeneous workload labels.
- Multiprocessing workload argument propagation.

For each selected challenge, explain:

Problem → Root Cause → Solution → Validation

Only describe an issue as solved if the implementation and tests confirm it.

Do not present an unresolved regression as a completed fix.

**Slide 8 — Experimental Methodology (45 seconds)**

Summarize:

- Datasets and workload categories.
- Model and tokenizer.
- Block size.
- Cache capacity.
- Arrival methodology.
- Concurrency.
- Random seeds.
- Baseline policies.

Define the primary metric:

Token-weighted prefix sharing rate = total cached prompt tokens / total prompt tokens.

Explain that the project evaluates simulated cache behavior rather than actual GPU performance.

**Slide 9 — Milestone 2: Single-Turn vs. Multi-Turn (65 seconds)**

Use newly validated or recovered ShareGPT results.

Show:

- Aggregate prefix sharing rate.
- Per-request sharing fraction CDF.
- Block hit-count distribution.
- Reuse interval distribution.

Select the most informative figures for the main slide and move supplementary figures to backup slides.

Explain the observed differences between single-turn and multi-turn conversations.

**Slide 10 — Milestone 2: Cache Parameter Sensitivity (55 seconds)**

Show the effect of:

- Cache capacity.
- Block size.
- Eviction policy.

Use readable line charts or grouped bar charts.

Explain the most important trade-offs.

Use only validated experimental results.

**Slide 11 — Milestone 3 Task 1: Heterogeneous Workloads (70 seconds)**

Show the comparison of LRU, LFU, and FIFO across all eight workload sources.

Highlight the strongest verified findings.

Historical reference observations include:

- Coding-agent workloads favor LRU over LFU.
- Bailian production traces show stronger LFU results.
- QuALITY shows strong FIFO performance.
- Some workloads exhibit nearly identical results across policies.

Update all numerical values using validated raw results.

Do not describe small differences as statistically significant without supporting evidence.

Explain how workload-dependent temporal locality may influence eviction behavior.

**Slide 12 — Milestone 3 Task 2: Custom Eviction Policies (65 seconds)**

Explain:

- Why a single baseline policy may not work well for mixed workloads.
- Workload-aware protection.
- Hit-aware protection.
- Adaptive reuse-aware protection.

Introduce the actual implementations of:

- TLRU.
- TFIFO.
- HLRU.
- HFIFO.
- RFIFO.

Use a concise comparison table or algorithm diagram.

Explain the most important design differences rather than presenting lengthy source code.

**Slide 13 — Mixed-Workload Results (80 seconds)**

Show validated results for:

- Skewed mixed workloads.
- Equal-token mixed workloads.
- Baseline versus custom policies.
- Per-workload sharing rates, if available.

Use a grouped bar chart comparing all eight policies.

Highlight:

- Best observed policy for each mixture.
- Improvement relative to the strongest baseline.
- Sensitivity to workload composition.
- Static versus adaptive policy behavior.

Do not claim that one policy is universally optimal.

If historical results cannot be reproduced or verified, clearly label them as preliminary or unavailable.

**Slide 14 — Reproducibility and Implementation Validation (50 seconds)**

The instructor explicitly states that TAs will run the code.

Provide a clear overview of:

- Environment setup.
- Dependency installation.
- Simulator startup.
- Workload preparation.
- Test execution.
- Running a representative experiment.
- Reproducing figures and tables.

Show short, verified command examples.

Reference the complete README for detailed instructions.

Include the latest actual test results.

Do not invent commands or report tests that were not executed.

**Slide 15 — Conclusion and Remaining Work (40 seconds)**

Summarize:

- Main simulator contribution.
- Homogeneous workload findings.
- Cache parameter trade-offs.
- Heterogeneous workload findings.
- Mixed-workload policy findings.
- Remaining limitations or future work.

Finish with three concise research takeaways.

Do not exaggerate results.

### 9.4 Backup Slides

Create supplementary slides for potential instructor questions.

Suggested topics:

- Detailed simulator execution path.
- Request-ID and trace matching.
- Virtual KV cache capacity calculation.
- Complete experimental configuration table.
- Full Milestone 2 results.
- Full Task 1 numerical results.
- Full Task 2 numerical results.
- Per-workload hit rates.
- Custom policy pseudocode.
- Multiprocessing regression and validation.
- Test coverage.
- Exact reproduction commands.
- AI Assistance disclosure.

Backup slides should not be included in the main presentation timing.

### 9.5 Presentation Design Requirements

Use a clean academic research presentation style.

- 16:9 widescreen layout.
- White or light neutral background.
- Consistent typography.
- Large readable titles and body text.
- Consistent colors for each eviction policy across all figures.
- Publication-quality diagrams and charts.
- Clear axis labels and units.
- Minimal decorative elements.
- No dense paragraphs.
- No unnecessary stock images.
- No screenshots of lengthy source code.
- No unnecessary audit terminology.

Use editable PowerPoint elements for text, tables, and diagrams wherever practical.

Use vector figures or high-resolution images for experimental plots.

Do not simply convert PDF pages into slide images.

Keep numerical results consistent with the final technical report.

Generate figures directly from validated experimental data.

### 9.6 English Speaker Notes

Write natural English speaker notes for every main slide.

The notes must:

- Explain the slide rather than merely repeat its text.
- Provide clear transitions between slides.
- Explain technical terminology when necessary.
- Explain why each experiment was performed.
- Highlight the main findings.
- Distinguish observations from hypotheses.
- Avoid unsupported claims.
- Be suitable for a graduate-level computer science presentation.

Target approximately 1,450–1,600 spoken English words in total.

Estimate speaking time using a reasonable speaking rate.

Keep the complete presentation below 15 minutes.

Provide a separate speaker script with:

- Slide number.
- Slide title.
- Estimated speaking time.
- English narration.
- Transition to the next slide.

Do not assign slides to specific presenters unless confirmed by the team.

### 9.7 Instructor Q&A Preparation

Prepare a separate Q&A document containing approximately 20 likely technical questions and concise answers.

Include questions such as:

- Why can prefix sharing be studied without actual GPU inference?
- Which parts of vLLM remain unchanged?
- How does the simulator generate output tokens?
- How is continuous batching preserved?
- How are duplicate prompts handled?
- How is virtual KV cache capacity calculated?
- Why is token-weighted sharing rate used?
- Why does multi-turn conversation behavior differ from single-turn behavior?
- How does block size affect prefix reuse?
- Why does LRU outperform LFU on coding-agent workloads?
- Why might LFU perform better on production traces?
- How are mixed workloads constructed?
- Why distinguish equal-token from equal-request mixtures?
- How do custom policies decide which blocks to protect?
- What happens when protected blocks fill the cache?
- How does RFIFO adapt to changing workload composition?
- How are workload labels obtained?
- Does the multiprocessing issue affect previous experiments?
- Can the results be generalized to actual GPU serving performance?
- How can the TAs reproduce the results?

Base answers on the actual implementation and verified experimental evidence.

### 9.8 TA Reproduction Instructions

Update the repository README with a tested, step-by-step reproduction guide.

Include:

1. Supported operating system and environment.
2. Python and dependency versions.
3. vLLM setup.
4. Required datasets and download instructions.
5. Trace preprocessing.
6. Simulator startup.
7. Unit and integration test commands.
8. A small smoke-test experiment.
9. Milestone 2 reproduction commands.
10. Milestone 3 Task 1 reproduction commands.
11. Milestone 3 Task 2 reproduction commands.
12. Figure and table generation.
13. Expected output directories.
14. Troubleshooting.

Provide a lightweight smoke-test path that the TAs can execute without running the complete experimental sweep.

Test the documented commands in a clean or isolated environment where practical.

Document any commands that could not be independently verified.

Do not require GPU hardware for the CPU-based simulation workflow unless an actual dependency makes this unavoidable.

### 9.9 Presentation Deliverables

Produce:

1. `Prefix_Sharing_Final_Presentation.pptx`
2. `Prefix_Sharing_Final_Presentation.pdf`
3. `Presentation_Speaker_Notes.md`
4. `Presentation_QA.md`
5. Updated `README.md`
6. Editable architecture and workflow diagrams.
7. Validated experimental figures.
8. Presentation timing summary.

Render every slide and visually inspect the output.

Verify:

- No text overflow.
- No clipped figures.
- No unreadable chart labels.
- No overlapping elements.
- Consistent author information.
- Consistent experimental numbers.
- Coverage of every instructor requirement.
- Total presentation duration below 15 minutes.

Preserve the original presentation.

Update the presentation after the final experimental results are available.

The final PowerPoint, technical report, README, experimental artifacts, and speaker notes must describe the same verified project state.
```

