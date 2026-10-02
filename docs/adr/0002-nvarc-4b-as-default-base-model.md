# NVARC's SFT'd Qwen3-4B as the default base model

We default the solver's base model to the ARC Prize 2025 winner's published
model (`sorokin/qwen3_4b_grids15_sft139`, a grid-SFT'd Qwen3-4B) rather than our
own Qwen2.5-Coder-7B, because a 7B in fp16 fills ~13 GB of a T4's 14.5 GB and
leaves no room for per-task TTT's LoRA activations — every candidate eval OOM'd
on a T4 — whereas the 4B (~8 GB) fits with real TTT headroom and is the exact
model that reached 24% on the 2025 private set.

## Consequences

- Adapters are base-specific: a LoRA trained on the 7B cannot load on the 4B, so
  switching the base invalidates any staged adapter trained against the other.
- We depend on a third party's model. Its license is assumed permissive (the
  prize required winners to open-source), recorded as an assumption to verify in
  `ASSUMPTIONS.md` before relying on it for our own prize claim.
