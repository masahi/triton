from triton.runtime import driver
from triton.runtime.jit import constexpr_function

__all__ = ["current_target"]


def current_target():
    try:
        active_driver = driver.active
    except RuntimeError:
        # If there is no active driver, return None
        return None
    return active_driver.get_current_target()


current_target.__triton_builtin__ = True


@constexpr_function
def is_cuda():
    target = current_target()
    return target is not None and target.backend == "cuda"


@constexpr_function
def cuda_capability_geq(major, minor=0):
    """
    DEPRECATED: Use has_feature() from triton.language.target_features instead.

    This function incorrectly assumes compute capability is monotonic.
    For example, sm120 (cc=120) lacks features present in sm100 (cc=100),
    such as MMAv5, TMA scatter/gather, and native MXFP4.

    Migration examples:
      cuda_capability_geq(10, 0) for TMA      → has_feature(TargetFeature.TMA_GATHER)
      cuda_capability_geq(10, 0) for MXFP4    → has_feature(TargetFeature.NATIVE_MXFP4)
      cuda_capability_geq(10, 0) for MMAv5    → has_feature(TargetFeature.MMA_V5)
      cuda_capability_geq(9, 0)  for MMAv3    → has_feature(TargetFeature.MMA_V3)
      cuda_capability_geq(7, 5)  for ldmatrix → has_feature(TargetFeature.LD_MATRIX)

    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    # Note: We don't emit a Python warning here because this is a constexpr
    # function that runs at JIT compile time, not at Python execution time.
    # Users should see deprecation notices in documentation and code comments.

    target = current_target()
    if target is None or target.backend != "cuda":
        return False
    assert isinstance(target.arch, int)
    return target.arch >= major * 10 + minor


@constexpr_function
def is_hip():
    target = current_target()
    return target is not None and target.backend == "hip"


@constexpr_function
def is_hip_cdna3():
    target = current_target()
    return target is not None and target.arch == "gfx942"


@constexpr_function
def is_hip_cdna4():
    target = current_target()
    return target is not None and target.arch == "gfx950"
