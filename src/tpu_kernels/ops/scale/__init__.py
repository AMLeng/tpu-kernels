"""Scale-by-2 op. Hello-world used to validate the harness end-to-end.

Pure memory-bound (1 flop/element, 2 bytes/element read+write at bf16).
Speed-of-light is HBM bandwidth — bench should approach 100% BW%.
"""

from tpu_kernels.ops.scale.naive import scale as scale_naive
from tpu_kernels.ops.scale.pallas import scale as scale_pallas
from tpu_kernels.ops.scale.xla import scale as scale_xla

__all__ = ["scale_naive", "scale_pallas", "scale_xla"]
