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
