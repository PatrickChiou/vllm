# Requirements-to-evidence checklist

Status applies to the supplied final source and raw-validated artifacts. "Completed and verified" is scoped to the listed evidence, not every upstream GPU feature.

| Requirement | Status | Evidence and limits |
|---|---|---|
| M1 CPU vLLM v0 online simulator | Completed and verified | Active scheduler/sequence/block path, trace executor, engine/API tests; unsupported extensions are documented |
| M2 Task1 causal client simulator | Completed and verified | Preparation, exact request IDs/tokens, causal replay, timing/failure tests and new matched workloads |
| M2 Task2 single/multi ShareGPT | Completed and verified | Matched200 conversations,200/608 requests per cell; LRU/block16 comparisons reuse the sensitivity sweep |
| M2 Task3 parameter sensitivity | Completed and verified |36 cells ×3 arrival seeds =108 runs; .05/.10GiB, blocks8/16/32, LRU/LFU/FIFO; bytes approximately fixed across blocks |
| M3 Task1 separate heterogeneous sources | Completed and verified |51 reused raw runs, eight sources, 21,834 requests; mean/SD/n/best gaps and complete source CDF supplement |
| M3 Task2 mixed policies | Completed and verified (required comparison) |16 new runs, two mixtures ×8 policies, seed0; identical within-mixture input/capacity; n=1 uncertainty unknown |
| Correctness and former multiprocessing issue | Completed and verified |146/0 applicable tests; optional label pickle/backend and both HTTP/frontend checks; B905 targeted lint passes |
| Experimental arithmetic/cohort integrity | Completed and verified |175 configurations; complete IDs, labels, causal order, hashes, counts, finite/bounded metrics and common policy controls checked |
| Reproducibility | Completed and verified within observed environment | Exact commands, versions, hashes, raw logs, resumable driver, tested smoke path and portable figure input resolution; fresh OS installation not repeated |
| Technical report | Completed and verified |Official MLSys2025 style, 6 main pages, confirmed authors/order/emails; compiled and all pages rendered/visually checked |
| AI disclosure and prompts | Completed and verified |Dedicated report section and appendix; code/test/script/figure assistance disclosed, no inferred human review |
| Final presentation | Completed and verified |15 main +9 backup slides; native editable PPTX and Impress PDF; all slides rendered/visually checked |
| Speaker notes/Q&A/timing | Completed and verified as generated artifacts |1589 spoken words including transitions,810-second plan,20 Q&A; timing is an estimate, not a human rehearsal |
| Team contribution/history evidence | Completed and verified within available history |Full original project history, baseline ancestry and diffs; confirmed identities/order; no effort percentages or presenter split |
| Mixed three-seed repetitions | Partially completed |One complete seed per mixture; extra repetitions not run ahead of required families; no mixed SD/CI |
| Optional robustness/ablations | Missing (optional) |No window/threshold/hold/cache-ratio ablation claimed |
| Protection/promotion counters | Missing (optional if unavailable) |New mixed evictions measured; promotion counters absent. M2/old Task1 evictions null |
| Human review and rehearsal | Not verified |Team must confirm understanding/review and rehearse in the presentation environment |

No required comparison remains blocked. Original artifacts are historical snapshots; the dated final tree supersedes their older evidence gaps only where the new logs support it. No remote push occurred.
