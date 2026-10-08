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
| 2026-07-15 | - | - | - | train COMPLETE but adapter dataset staging FAILED; launched explore[poe] commit v1 |
| 2026-07-16 | - | - | - | explore competition_submit FAILED for v1; launched backlog[poe_regate] (eval) v2 |
| 2026-07-16 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v2 status=RUNNING |
| 2026-07-17 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v2 status=CANCEL; launched explore[ttt96] commit v3 |
| 2026-07-17 | - | - | - | explore competition_submit FAILED for v3; launched backlog[ttt_steps_sweep] (eval) v4 |
| 2026-07-18 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v4 status=RUNNING |
| 2026-07-18 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v4 status=CANCEL; launched explore[dfs] commit v5 |
| 2026-07-19 | - | - | - | pending_lb: submission 54830777 has no numeric score yet; pushed eval v6 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-07-19 | - | - | - | pending_lb: submission 54830777 has no numeric score yet; FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v6 status=ERROR |
| 2026-07-19 | 54830777 | no-adapter selection=votes | 0.83 | promoted |
| 2026-07-19 | - | - | - | pending_lb resolved: ref=54830777 score=0.83 -> promoted; pushed eval v7 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-07-20 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v7 status=ERROR |
| 2026-07-20 | - | - | - | pushed eval v8 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-07-21 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v8 status=ERROR |
| 2026-07-21 | - | - | - | pushed eval v9 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-07-22 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v9 status=ERROR |
| 2026-07-22 | - | - | - | pushed eval v10 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-07-23 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v10 status=ERROR |
| 2026-07-23 | - | - | - | pushed eval v11 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-07-24 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v11 status=ERROR |
| 2026-07-24 | - | - | - | pushed eval v12 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-07-25 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v12 status=ERROR |
| 2026-07-25 | - | - | - | pushed eval v13 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-07-26 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v13 status=ERROR |
| 2026-07-26 | - | - | - | pushed eval v14 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-07-27 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v14 status=ERROR |
| 2026-07-27 | - | - | - | pushed eval v15 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-07-28 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v15 status=ERROR |
| 2026-07-28 | - | - | - | pushed eval v16 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-07-29 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v16 status=ERROR |
| 2026-07-29 | - | - | - | pushed eval v17 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-07-30 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v17 status=ERROR |
| 2026-07-30 | - | - | - | pushed eval v18 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-07-31 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v18 status=ERROR |
| 2026-07-31 | - | - | - | pushed eval v19 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-01 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v19 status=ERROR |
| 2026-08-01 | - | - | - | pushed eval v20 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-02 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v20 status=ERROR |
| 2026-08-02 | - | - | - | pushed eval v21 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-02 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v21 status=RUNNING |
| 2026-08-02 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v21 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 15.56 GiB. GPU 0 has a total capacity of 14.56 GiB of which 1.61 GiB is free. Including non-PyTorch memory, this process has 12.95 GiB memory in use. Of the allocated memory 9.05 GiB is allocated by PyTorch, and 3.77 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fai |
| 2026-08-03 | - | - | - | pushed eval v22 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-03 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v22 status=RUNNING |
| 2026-08-04 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v22 status=CANCEL |
| 2026-08-04 | - | - | - | pushed eval v23 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-05 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v23 status=RUNNING |
| 2026-08-05 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v23 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.73 GiB. GPU 0 has a total capacity of 14.56 GiB of which 2.14 GiB is free. Including non-PyTorch memory, this process has 12.42 GiB memory in use. Of the allocated memory 9.88 GiB is allocated by PyTorch, and 2.40 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fail |
| 2026-08-06 | - | - | - | pushed eval v24 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-06 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v24 status=RUNNING |
| 2026-08-07 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v24 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 15.56 GiB. GPU 0 has a total capacity of 14.56 GiB of which 1.61 GiB is free. Including non-PyTorch memory, this process has 12.95 GiB memory in use. Of the allocated memory 9.05 GiB is allocated by PyTorch, and 3.77 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fai |
| 2026-08-07 | - | - | - | pushed eval v25 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-08 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v25 status=RUNNING |
| 2026-08-08 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v25 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.79 GiB. GPU 0 has a total capacity of 14.56 GiB of which 2.14 GiB is free. Including non-PyTorch memory, this process has 12.42 GiB memory in use. Of the allocated memory 9.95 GiB is allocated by PyTorch, and 2.34 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fail |
| 2026-08-09 | - | - | - | pushed eval v26 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-09 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v26 status=RUNNING |
| 2026-08-10 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v26 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 570.00 MiB. GPU 1 has a total capacity of 14.56 GiB of which 532.81 MiB is free. Including non-PyTorch memory, this process has 14.04 GiB memory in use. Of the allocated memory 13.28 GiB is allocated by PyTorch, and 645.91 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_t |
| 2026-08-10 | - | - | - | pushed eval v27 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-11 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v27 status=RUNNING |
| 2026-08-11 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v27 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.13 GiB. GPU 0 has a total capacity of 14.56 GiB of which 2.14 GiB is free. Including non-PyTorch memory, this process has 12.42 GiB memory in use. Of the allocated memory 10.34 GiB is allocated by PyTorch, and 1.95 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fai |
| 2026-08-12 | - | - | - | pushed eval v28 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-12 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v28 status=RUNNING |
| 2026-08-13 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v28 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 15.56 GiB. GPU 0 has a total capacity of 14.56 GiB of which 1.61 GiB is free. Including non-PyTorch memory, this process has 12.95 GiB memory in use. Of the allocated memory 9.05 GiB is allocated by PyTorch, and 3.77 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fai |
| 2026-08-13 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-08-14 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-08-14 | - | - | - | pushed eval v29 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-15 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v29 status=CANCEL |
| 2026-08-15 | - | - | - | pushed eval v30 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-16 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v30 status=RUNNING |
| 2026-08-16 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v30 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.13 GiB. GPU 0 has a total capacity of 14.56 GiB of which 2.14 GiB is free. Including non-PyTorch memory, this process has 12.42 GiB memory in use. Of the allocated memory 10.34 GiB is allocated by PyTorch, and 1.95 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fai |
| 2026-08-17 | - | - | - | pushed eval v31 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-17 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v31 status=RUNNING |
| 2026-08-18 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v31 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.28 GiB. GPU 0 has a total capacity of 14.56 GiB of which 1.61 GiB is free. Including non-PyTorch memory, this process has 12.95 GiB memory in use. Of the allocated memory 7.87 GiB is allocated by PyTorch, and 4.95 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fail |
| 2026-08-18 | - | - | - | pushed eval v32 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-19 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v32 status=RUNNING |
| 2026-08-19 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v32 status=CANCEL |
| 2026-08-20 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-08-20 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-08-21 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-08-21 | - | - | - | pushed eval v33 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-22 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v33 status=RUNNING |
| 2026-08-22 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v33 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.43 GiB. GPU 0 has a total capacity of 14.56 GiB of which 1.61 GiB is free. Including non-PyTorch memory, this process has 12.95 GiB memory in use. Of the allocated memory 8.30 GiB is allocated by PyTorch, and 4.51 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fail |
| 2026-08-23 | - | - | - | pushed eval v34 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-23 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v34 status=RUNNING |
| 2026-08-24 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v34 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.83 GiB. GPU 0 has a total capacity of 14.56 GiB of which 2.14 GiB is free. Including non-PyTorch memory, this process has 12.42 GiB memory in use. Of the allocated memory 10.14 GiB is allocated by PyTorch, and 2.15 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fai |
| 2026-08-24 | - | - | - | pushed eval v35 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-25 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v35 status=RUNNING |
| 2026-08-25 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v35 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 570.00 MiB. GPU 1 has a total capacity of 14.56 GiB of which 532.81 MiB is free. Including non-PyTorch memory, this process has 14.04 GiB memory in use. Of the allocated memory 13.28 GiB is allocated by PyTorch, and 645.91 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_t |
| 2026-08-26 | - | - | - | pushed eval v36 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-26 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v36 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.96 GiB. GPU 0 has a total capacity of 14.56 GiB of which 1.61 GiB is free. Including non-PyTorch memory, this process has 12.95 GiB memory in use. Of the allocated memory 8.11 GiB is allocated by PyTorch, and 4.71 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fail |
| 2026-08-27 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-08-28 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-08-28 | - | - | - | pushed eval v37 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-29 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v37 status=RUNNING |
| 2026-08-29 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v37 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.79 GiB. GPU 0 has a total capacity of 14.56 GiB of which 2.14 GiB is free. Including non-PyTorch memory, this process has 12.42 GiB memory in use. Of the allocated memory 9.95 GiB is allocated by PyTorch, and 2.34 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fail |
| 2026-08-30 | - | - | - | pushed eval v38 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-08-30 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v38 status=RUNNING |
| 2026-08-31 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v38 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.92 GiB. GPU 0 has a total capacity of 14.56 GiB of which 2.14 GiB is free. Including non-PyTorch memory, this process has 12.42 GiB memory in use. Of the allocated memory 8.38 GiB is allocated by PyTorch, and 3.90 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fail |
| 2026-08-31 | - | - | - | pushed eval v39 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-09-01 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v39 status=CANCEL |
| 2026-09-01 | - | - | - | pushed eval v40 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-09-02 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v40 status=RUNNING |
| 2026-09-02 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v40 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.96 GiB. GPU 0 has a total capacity of 14.56 GiB of which 1.61 GiB is free. Including non-PyTorch memory, this process has 12.95 GiB memory in use. Of the allocated memory 8.11 GiB is allocated by PyTorch, and 4.71 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fail |
| 2026-09-03 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-03 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-04 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-05 | - | - | - | pushed eval v42 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-09-05 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v42 status=RUNNING |
| 2026-09-06 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v42 status=CANCEL |
| 2026-09-06 | - | - | - | pushed eval v43 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-09-07 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v43 status=RUNNING |
| 2026-09-07 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v43 status=CANCEL |
| 2026-09-08 | - | - | - | pushed eval v44 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-09-08 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v44 status=RUNNING |
| 2026-09-09 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v44 status=CANCEL |
| 2026-09-09 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-10 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-10 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-11 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-12 | - | - | - | pushed eval v46 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-09-12 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v46 status=RUNNING |
| 2026-09-13 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v46 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 15.56 GiB. GPU 0 has a total capacity of 14.56 GiB of which 1.61 GiB is free. Including non-PyTorch memory, this process has 12.95 GiB memory in use. Of the allocated memory 9.05 GiB is allocated by PyTorch, and 3.77 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fai |
| 2026-09-13 | - | - | - | pushed eval v47 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-09-14 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v47 status=RUNNING |
| 2026-09-15 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v47 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 15.56 GiB. GPU 0 has a total capacity of 14.56 GiB of which 1.61 GiB is free. Including non-PyTorch memory, this process has 12.95 GiB memory in use. Of the allocated memory 9.05 GiB is allocated by PyTorch, and 3.77 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fai |
| 2026-09-15 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-16 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-16 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-17 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-17 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-18 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-18 | - | - | - | pushed eval v48 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-09-19 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v48 status=RUNNING |
| 2026-09-19 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v48 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 15.56 GiB. GPU 0 has a total capacity of 14.56 GiB of which 1.61 GiB is free. Including non-PyTorch memory, this process has 12.95 GiB memory in use. Of the allocated memory 9.05 GiB is allocated by PyTorch, and 3.77 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fai |
| 2026-09-20 | - | - | - | pushed eval v49 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-09-20 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v49 status=RUNNING |
| 2026-09-21 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v49 status=CANCEL |
| 2026-09-21 | - | - | - | pushed eval v50 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-09-22 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v50 status=CANCEL |
| 2026-09-22 | - | - | - | pushed eval v51 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-09-23 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v51 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.79 GiB. GPU 0 has a total capacity of 14.56 GiB of which 2.14 GiB is free. Including non-PyTorch memory, this process has 12.42 GiB memory in use. Of the allocated memory 9.95 GiB is allocated by PyTorch, and 2.34 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_ttt fail |
| 2026-09-23 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-24 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-24 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-25 | - | - | - | eval push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-25 | - | - | - | pushed eval v52 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-09-26 | - | - | - | waiting: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v52 status=RUNNING |
| 2026-09-26 | - | - | - | FAILED: eval kernel sebmontreal/arc-agi-2-autopilot-submit-2 v52 status=CANCEL; error: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 570.00 MiB. GPU 1 has a total capacity of 14.56 GiB of which 532.81 MiB is free. Including non-PyTorch memory, this process has 14.04 GiB memory in use. Of the allocated memory 13.28 GiB is allocated by PyTorch, and 645.91 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) | ERROR:arc.pipeline:solver llm_t |
| 2026-09-27 | - | - | - | pushed eval v53 for candidate (sebmontreal/arc-agi-2-adapter-autopilot) |
| 2026-09-27 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push BUSY (2 GPU slots in use); retrying next tick; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-28 | - | - | - | backlog[nvarc_sft_regate] push FAILED (1/3); retrying next tick: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached.; explore[nvarc_sft] push FAILED: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached. |
| 2026-09-29 | - | - | - | backlog[nvarc_sft_regate] push deferred (capacity); retrying next tick: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached.; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-09-30 | - | - | - | backlog[nvarc_sft_regate] push FAILED (1/3); retrying next tick: Traceback (most recent call last):
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connection.py", line 198, in _new_conn
    sock = connection.create_connection(
        (self._dns_host, self.port),
    ...<2 lines>...
        socket_options=self.socket_options,
    )
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/util/connection.py", line 60, in create_connection
    for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/lib/python3.13/socket.py", line 977, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
               ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
socket.gaierror: [Errno 8] nodename nor servname provided, or not known

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connectionpool.py", line 787, in urlopen
    response = self._make_request(
        conn,
    ...<10 lines>...
        **response_kw,
    )
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connectionpool.py", line 488, in _make_request
    raise new_e
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connectionpool.py", line 464, in _make_request
    self._validate_conn(conn)
    ~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connectionpool.py", line 1093, in _validate_conn
    conn.connect()
    ~~~~~~~~~~~~^^
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connection.py", line 753, in connect
    self.sock = sock = self._new_conn()
                       ~~~~~~~~~~~~~~^^
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connection.py", line 205, in _new_conn
    raise NameResolutionError(self.host, self, e) from e
urllib3.exceptions.NameResolutionError: <urllib3.connection.HTTPSConnection object at 0x105d10ad0>: Failed to resolve 'api.kaggle.com' ([Errno 8] nodename nor servname provided, or not known)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/opt/anaconda3/lib/python3.13/site-packages/requests/adapters.py", line 696, in send
    resp = conn.urlopen(
        method=request.method,
    ...<9 lines>...
        chunked=chunked,
    )
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connectionpool.py", line 841, in urlopen
    retries = retries.increment(
        method, url, error=new_e, _pool=self, _stacktrace=sys.exc_info()[2]
    )
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/util/retry.py", line 519, in increment
    raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='api.kaggle.com', port=443): Max retries exceeded with url: /v1/security.OAuthService/IntrospectToken (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x105d10ad0>: Failed to resolve 'api.kaggle.com' ([Errno 8] nodename nor servname provided, or not known)"))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/anaconda3/bin/kaggle", line 3, in <module>
    from kaggle.cli import main
  File "/opt/anaconda3/lib/python3.13/site-packages/kaggle/__init__.py", line 8, in <module>
    api.authenticate()
    ~~~~~~~~~~~~~~~~^^
  File "/opt/anaconda3/lib/python3.13/site-packages/kaggle/api/kaggle_api_extended.py", line 1062, in authenticate
    if self._authenticate_with_access_token():
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/opt/anaconda3/lib/python3.13/site-packages/kaggle/api/kaggle_api_extended.py", line 1099, in _authenticate_with_access_token
    username = self._introspect_token(access_token)
  File "/opt/anaconda3/lib/python3.13/site-packages/kaggle/api/kaggle_api_extended.py", line 1135, in _introspect_token
    response = kaggle.security.oauth_client.introspect_token(request)
  File "/opt/anaconda3/lib/python3.13/site-packages/kagglesdk/security/services/oauth_service.py", line 44, in introspect_token
    return self._client.call("security.OAuthService", "IntrospectToken", request, IntrospectTokenResponse)
           ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/lib/python3.13/site-packages/kagglesdk/kaggle_http_client.py", line 104, in call
    http_response = self._session.send(http_request, **settings)
  File "/opt/anaconda3/lib/python3.13/site-packages/requests/sessions.py", line 784, in send
    r = adapter.send(request, **kwargs)
  File "/opt/anaconda3/lib/python3.13/site-packages/requests/adapters.py", line 729, in send
    raise ConnectionError(e, request=request)
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.kaggle.com', port=443): Max retries exceeded with url: /v1/security.OAuthService/IntrospectToken (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x105d10ad0>: Failed to resolve 'api.kaggle.com' ([Errno 8] nodename nor servname provided, or not known)")); explore[nvarc_sft] push FAILED: Traceback (most recent call last):
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connection.py", line 198, in _new_conn
    sock = connection.create_connection(
        (self._dns_host, self.port),
    ...<2 lines>...
        socket_options=self.socket_options,
    )
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/util/connection.py", line 60, in create_connection
    for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/lib/python3.13/socket.py", line 977, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
               ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
socket.gaierror: [Errno 8] nodename nor servname provided, or not known

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connectionpool.py", line 787, in urlopen
    response = self._make_request(
        conn,
    ...<10 lines>...
        **response_kw,
    )
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connectionpool.py", line 488, in _make_request
    raise new_e
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connectionpool.py", line 464, in _make_request
    self._validate_conn(conn)
    ~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connectionpool.py", line 1093, in _validate_conn
    conn.connect()
    ~~~~~~~~~~~~^^
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connection.py", line 753, in connect
    self.sock = sock = self._new_conn()
                       ~~~~~~~~~~~~~~^^
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connection.py", line 205, in _new_conn
    raise NameResolutionError(self.host, self, e) from e
urllib3.exceptions.NameResolutionError: <urllib3.connection.HTTPSConnection object at 0x107e10ad0>: Failed to resolve 'api.kaggle.com' ([Errno 8] nodename nor servname provided, or not known)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/opt/anaconda3/lib/python3.13/site-packages/requests/adapters.py", line 696, in send
    resp = conn.urlopen(
        method=request.method,
    ...<9 lines>...
        chunked=chunked,
    )
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/connectionpool.py", line 841, in urlopen
    retries = retries.increment(
        method, url, error=new_e, _pool=self, _stacktrace=sys.exc_info()[2]
    )
  File "/opt/anaconda3/lib/python3.13/site-packages/urllib3/util/retry.py", line 519, in increment
    raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='api.kaggle.com', port=443): Max retries exceeded with url: /v1/security.OAuthService/IntrospectToken (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x107e10ad0>: Failed to resolve 'api.kaggle.com' ([Errno 8] nodename nor servname provided, or not known)"))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/anaconda3/bin/kaggle", line 3, in <module>
    from kaggle.cli import main
  File "/opt/anaconda3/lib/python3.13/site-packages/kaggle/__init__.py", line 8, in <module>
    api.authenticate()
    ~~~~~~~~~~~~~~~~^^
  File "/opt/anaconda3/lib/python3.13/site-packages/kaggle/api/kaggle_api_extended.py", line 1062, in authenticate
    if self._authenticate_with_access_token():
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/opt/anaconda3/lib/python3.13/site-packages/kaggle/api/kaggle_api_extended.py", line 1099, in _authenticate_with_access_token
    username = self._introspect_token(access_token)
  File "/opt/anaconda3/lib/python3.13/site-packages/kaggle/api/kaggle_api_extended.py", line 1135, in _introspect_token
    response = kaggle.security.oauth_client.introspect_token(request)
  File "/opt/anaconda3/lib/python3.13/site-packages/kagglesdk/security/services/oauth_service.py", line 44, in introspect_token
    return self._client.call("security.OAuthService", "IntrospectToken", request, IntrospectTokenResponse)
           ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/lib/python3.13/site-packages/kagglesdk/kaggle_http_client.py", line 104, in call
    http_response = self._session.send(http_request, **settings)
  File "/opt/anaconda3/lib/python3.13/site-packages/requests/sessions.py", line 784, in send
    r = adapter.send(request, **kwargs)
  File "/opt/anaconda3/lib/python3.13/site-packages/requests/adapters.py", line 729, in send
    raise ConnectionError(e, request=request)
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.kaggle.com', port=443): Max retries exceeded with url: /v1/security.OAuthService/IntrospectToken (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x107e10ad0>: Failed to resolve 'api.kaggle.com' ([Errno 8] nodename nor servname provided, or not known)")) |
| 2026-10-02 | - | - | - | backlog[nvarc_sft_regate] push deferred (capacity); retrying next tick: Kernel push error: Maximum weekly GPU quota of 45.00 hours reached.; explore[nvarc_sft] push BUSY; retrying next tick |
| 2026-10-02 | - | - | - | launched backlog[nvarc_sft_regate] (eval) v54 |
| 2026-10-03 | - | - | - | eval COMPLETE but no summary parsed; discarding candidate; launched explore[nvarc_sft] commit v55 |
| 2026-10-03 | - | - | - | SUBMITTED (exploration): sebmontreal/arc-agi-2-autopilot-submit-2:v55:2026-10-03; launched backlog[dfs_regate] (eval) v56 |
| 2026-10-04 | sebmontreal/arc-agi-2-autopilot-submit-2:v55:2026-10-03 | no-adapter selection=votes | 0.0 | rejected |
| 2026-10-04 | - | - | - | pending_lb resolved: ref=sebmontreal/arc-agi-2-autopilot-submit-2:v55:2026-10-03 score=0.0 -> rejected; eval COMPLETE: no improvement (0 vs best 0); discarded; launched explore[poe] commit v57 |
| 2026-10-04 | - | - | - | SUBMITTED (exploration): sebmontreal/arc-agi-2-autopilot-submit-2:v57:2026-10-04; launched backlog[adapter_hard_6000] (train) v2 |
| 2026-10-05 | sebmontreal/arc-agi-2-autopilot-submit-2:v57:2026-10-04 | no-adapter selection=poe | 0.0 | rejected |
| 2026-10-05 | - | - | - | pending_lb resolved: ref=sebmontreal/arc-agi-2-autopilot-submit-2:v57:2026-10-04 score=0.0 -> rejected; train COMPLETE: adapter staged -> sebmontreal/arc-agi-2-adapter-autopilot; candidate ready for eval |
| 2026-10-05 | - | - | - | eval push deferred (GPU capacity: slots busy or quota spent); retrying |
| 2026-10-06 | - | - | - | eval push deferred (GPU capacity: slots busy or quota spent); retrying |
| 2026-10-06 | - | - | - | eval push deferred (GPU capacity: slots busy or quota spent); retrying |
| 2026-10-07 | - | - | - | eval push deferred (GPU capacity: slots busy or quota spent); retrying |
| 2026-10-07 | - | - | - | eval push deferred (GPU capacity: slots busy or quota spent); retrying |
| 2026-10-08 | - | - | - | eval push deferred (GPU capacity: slots busy or quota spent); retrying |
