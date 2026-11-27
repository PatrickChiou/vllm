# Current requirements and evidence

| Requirement | Current evidence | Limits |
|---|---|---|
| Simulator/API correctness | Fresh full suite: 153/0 on ad1d826542 | Simulator scope, not GPU upstream suite |
| M2 and M3 Task1 | Historical raw-validated 108 + 51 runs | Prepared under earlier code; not rerun |
| Required two-mixture comparison | Historical raw-validated 16 runs, all eight policies | One seed; no SD/CI |
| Hold/budget/admission sensitivity | Upstream reports 125 cells, five mixtures; full rounded table integrated | Raw files unavailable, fixed order, single seed, HFIFO absent |
| Parameter robustness | Descriptive macro mean/regret from reported larger table | In-sample selection and rounding; no universal winner or held-out claim |
| New RFIFO controls/stop token fix | Source review and seven additional tests; full suite passes | Preparation changes can change future cohorts |
| Report/presentation | Updated English source, PDF and editable deck; fresh rendering checks | See evidence/document-checks.json for actual final counts |
| Reproduction | Fresh test command, source attribution, figure/document build scripts; upstream grid CLI reviewed | New full grid not run; resume completeness caveat |
| Optional counters | Historical mixed eviction counts retained | New sweep counters/promotion stats unavailable |
| Team attribution/AI | Thomas's update credited; Patrick's AI-assisted integration disclosed | No contribution percentages, presenter split or human-review assertion |
| Raw replay verification for new sweep | Not completed: raw records not supplied | Reported zero failures are attributed, not independently certified |
| Human rehearsal/GPU/fresh OS/extra seeds | Not verified | Not claimed |

The smaller-scale sweep is discussed as a separate reported experiment, not a repeat or interchangeable measurement. Original November 24 requirements/evidence remain a historical snapshot.
