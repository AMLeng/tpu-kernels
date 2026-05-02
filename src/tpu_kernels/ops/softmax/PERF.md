# Softmax

Target: ≥80% HBM BW @ (8192, 8192) bf16
Current: not measured yet (xla scaffold landed; pending TPU bench)
Bottleneck: TBD — `--dump-hlo` first to confirm XLA fuses the stable-form chain (max → sub → exp → sum → div) into one elementwise loop.
Next: bench `softmax_xla` on v5e; if it clears 80% BW under device timing, mark done — otherwise add `pallas.py`.
