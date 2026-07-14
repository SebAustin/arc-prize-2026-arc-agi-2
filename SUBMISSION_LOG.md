# Submission Log

Chronological record of every Kaggle competition submission (with public-LB
reconciliation) and autopilot tick status, appended by
`scripts/daily_autopilot.py`. Leader ≈ 55 (research frontier; this 7B+TTT
architecture realistically reaches single digits → low-teens — the autopilot
*reduces the gap*, it does not overtake the leader).

| Date | Ref | Config | LB Score | Note |
|------|-----|--------|----------|------|
| 2026-07-08 | 54474923 | v11 (7B + TTT-48 + 8-aug + likelihood) | 0.83 | baseline |
| 2026-07-12 | - | - | - | launched backlog[adapter_hard_2500] (train) v8 |
| 2026-07-12 | - | - | - | FAILED: train kernel sebmontreal/arc-agi-2-train-adapter-rung-4 v8 status=ERROR |
| 2026-07-13 | - | - | - | launched backlog[poe_regate] (eval) v13 |
| 2026-07-13 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-self-contained-submission-notebook v13 status=ERROR |
| 2026-07-13 | - | - | - | operator: widened eval canary 40 → full 120-task public split (sub-1% sensitivity); rewound backlog_index 2 → 0 so the fixed adapter_hard_2500 train launches next |
| 2026-07-14 | - | - | - | backlog[adapter_hard_2500] push FAILED: Your kernel title does not resolve to the specified id. This may result in surprising behavior. We suggest making your title something that resolves to the specified id. See https://en.wikipedia.org/wiki/Clean_URL#Slug for more information on how slugs are determined.

409 Client Error: Conflict for url: https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel |
| 2026-07-14 | - | - | - | operator: fixed Kaggle-API 409 (titles now slug-resolve to kernel ids), 404-status misparse, push-failure retry (3 strikes); added exploration-submission lane (untried variants use idle daily slots — Kaggle ranks best); rewound backlog_index 1 → 0 |
| 2026-07-14 | - | - | - | backlog[adapter_hard_2500] push BUSY (2 GPU slots in use); retrying next tick; explore[poe] push BUSY; retrying next tick |
| 2026-07-14 | - | - | - | backlog[adapter_hard_2500] push FAILED (1/3); retrying next tick: Kernel push error: Notebook not found; explore[poe] push FAILED: Kernel push error: Notebook not found |
| 2026-07-14 | - | - | - | launched backlog[adapter_hard_2500] (train) v1; explore[poe] push BUSY; retrying next tick |
| 2026-07-14 | - | - | - | waiting: train kernel sebmontreal/arc-agi-2-autopilot-train v1 status=RUNNING; explore[poe] push FAILED: Kernel push error: Notebook not found |
| 2026-07-14 | - | - | - | waiting: train kernel sebmontreal/arc-agi-2-autopilot-train v1 status=RUNNING; submission kernel id burned (create rejected — Kaggle busy-create bug); rotated to sebmontreal/arc-agi-2-autopilot-submit-2; retrying next tick |
