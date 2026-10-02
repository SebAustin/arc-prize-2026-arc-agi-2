# Optimize for the private rerun, not the public leaderboard

We deliberately do NOT optimize toward the public leaderboard score. The locally
visible `test_challenges.json` is 240 tasks that are all in the training set — a
public placeholder, swapped for the hidden set at the private rerun — and the
top public scores (83/76/75) are implausible against the 2025 private SOTA of
24%, consistent with public-task memorization we neither can nor should
replicate. The prize is decided on the private rerun, so our promotion signal is
the held-out public-evaluation canary (120 genuinely unseen tasks), not the
public leaderboard.

## Consequences

- A future reader will see our canary score sit near zero while the public board
  shows high numbers; this gap is expected, not a bug to "fix" by chasing the LB.
- Ungated exploration submissions are still worthwhile because the competition
  ranks an entrant's best submission, but they are never treated as the
  generalization signal.
