# Attention (no-cache, single-seq, fwd)

Target: ≥70% MFU @ (T=4096, N=16, H=128), bf16
Current: not measured yet (scaffold only — naive + xla; bench suite lands with the Pallas variant)
Bottleneck: xla materializes the (N, T, T) attention-scores matrix to HBM between the score and value matmuls; for long T the read+write of that intermediate dominates HBM traffic and the kernel falls off the roofline. The Pallas (flash) variant — streaming K/V tiles + online softmax + scratch-resident running statistics — is the answer.
Next: write the Pallas flash variant and stand up `benchmarks/suites/attention.py` so the wall (and the Pallas win) become measurable.
