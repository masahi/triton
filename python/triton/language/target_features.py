"""
Target feature detection API for Triton.

This module provides explicit feature queries to replace compute capability
comparisons. This is necessary because newer hardware (e.g., sm120) may have
fewer features than older hardware (e.g., sm100), breaking assumptions in
`cc >= X` checks.

Example:
    # OLD (incorrect for sm120):
    if cuda_capability_geq(10, 0):  # Assumes sm120 has all sm100 features
        use_tma_gather()

    # NEW (correct):
    if has_feature(TargetFeature.TMA_GATHER):  # Explicitly checks feature
        use_tma_gather()
"""

from enum import IntEnum, auto
from triton.runtime import driver
from triton.runtime.jit import constexpr_function

__all__ = ["TargetFeature", "has_feature", "get_best_mma_version"]


class TargetFeature(IntEnum):
    """Hardware features that can be queried across different target architectures."""

    # Memory operations
    LD_MATRIX = auto()          # Load matrix from shared memory (sm75+)
    ST_MATRIX = auto()          # Store matrix to shared memory (sm90+)
    LDST_MATRIX_B8 = auto()     # 8-bit ldstmatrix (sm100-sm119 only, NOT sm120)

    # TMA (Tensor Memory Accelerator)
    TMA_GENERAL = auto()        # Basic TMA support (sm90+)
    TMA_GATHER = auto()         # TMA gather operations (sm100-sm119 only, NOT sm120)
    TMA_SCATTER = auto()        # TMA scatter operations (sm100-sm119 only, NOT sm120)

    # MMA (Matrix Multiply-Accumulate) versions
    MMA_V1 = auto()             # MMA version 1 (sm70+)
    MMA_V2 = auto()             # MMA version 2 (sm75+, sm120+)
    MMA_V3 = auto()             # MMA version 3 (sm90-sm99 only)
    MMA_V5 = auto()             # MMA version 5 (sm100-sm119 only, NOT sm120)
    MMA_V2_FP8_NATIVE = auto()  # Native FP8 in MMAv2 (sm89, sm120 only)

    # Numeric types
    NATIVE_MXFP4 = auto()       # Native MXFP4 support (sm100-sm119 only, NOT sm120)
    NATIVE_FP8 = auto()         # Native FP8 support (sm89+)

    # Execution features
    VECTORIZED_ATOMICS = auto()               # Vectorized atomic operations
    INDEPENDENT_EPILOGUE_PIPELINING = auto()  # Independent epilogue pipelining (sm90+)

    # PTX instructions
    PTX_MIN_MAX_NAN = auto()    # PTX min/max with NaN handling
    PTX_CVT_E2M1 = auto()       # PTX cvt.e2m1 instruction (MXFP4)

    # AMD-specific features
    V_DOT_INSTRUCTION = auto()      # v_dot instruction
    DIRECT_TO_LDS_SCATTER = auto()  # Direct to LDS scatter
    LDS_TRANS_LOAD = auto()         # LDS transposed load


# Mapping from Python enum to C++ feature names (for future C++ integration)
_FEATURE_NAME_MAP = {
    TargetFeature.LD_MATRIX: "LdMatrix",
    TargetFeature.ST_MATRIX: "StMatrix",
    TargetFeature.LDST_MATRIX_B8: "LdStMatrixB8",
    TargetFeature.TMA_GENERAL: "TMAGeneral",
    TargetFeature.TMA_GATHER: "TMAGather",
    TargetFeature.TMA_SCATTER: "TMAScatter",
    TargetFeature.MMA_V1: "MMAv1",
    TargetFeature.MMA_V2: "MMAv2",
    TargetFeature.MMA_V3: "MMAv3",
    TargetFeature.MMA_V5: "MMAv5",
    TargetFeature.MMA_V2_FP8_NATIVE: "MMAv2FP8Native",
    TargetFeature.NATIVE_MXFP4: "NativeMXFP4",
    TargetFeature.NATIVE_FP8: "NativeFP8",
    TargetFeature.VECTORIZED_ATOMICS: "VectorizedAtomics",
    TargetFeature.INDEPENDENT_EPILOGUE_PIPELINING: "IndependentEpiloguePipelining",
    TargetFeature.PTX_MIN_MAX_NAN: "PTXMinMaxNaN",
    TargetFeature.PTX_CVT_E2M1: "PTXCvtE2M1",
    TargetFeature.V_DOT_INSTRUCTION: "VDotInstruction",
    TargetFeature.DIRECT_TO_LDS_SCATTER: "DirectToLDSScatter",
    TargetFeature.LDS_TRANS_LOAD: "LDSTransLoad",
}


def _get_nvidia_features(compute_capability: int) -> set:
    """
    Internal function to determine NVIDIA features from compute capability.
    This mirrors the C++ implementation in TargetArchitecture.cpp.
    """
    features = set()
    cc = compute_capability

    # Memory operations
    if cc >= 75:
        features.add(TargetFeature.LD_MATRIX)
    if cc >= 90:
        features.add(TargetFeature.ST_MATRIX)
    # LdStMatrixB8: Only sm100-sm119 (NOT sm120+)
    if cc >= 100 and cc < 120:
        features.add(TargetFeature.LDST_MATRIX_B8)

    # TMA (Tensor Memory Accelerator)
    if cc >= 90:
        features.add(TargetFeature.TMA_GENERAL)
    # TMA Gather/Scatter: Only sm100-sm119 (NOT sm120+)
    if cc >= 100 and cc < 120:
        features.add(TargetFeature.TMA_GATHER)
        features.add(TargetFeature.TMA_SCATTER)

    # MMA versions
    if cc >= 70:
        features.add(TargetFeature.MMA_V1)
    if cc >= 75:
        features.add(TargetFeature.MMA_V2)
    # MMAv3: sm90-sm99 only
    if cc >= 90 and cc < 100:
        features.add(TargetFeature.MMA_V3)
    # MMAv5: sm100-sm119 only (NOT sm120+)
    if cc >= 100 and cc < 120:
        features.add(TargetFeature.MMA_V5)

    # Special: Native FP8 in MMAv2 for sm89 and sm120
    if cc == 89 or cc == 120:
        features.add(TargetFeature.MMA_V2_FP8_NATIVE)

    # Native MXFP4: sm100-sm119 only (NOT sm120+)
    if cc >= 100 and cc < 120:
        features.add(TargetFeature.NATIVE_MXFP4)

    # Native FP8 (general)
    if cc >= 89:
        features.add(TargetFeature.NATIVE_FP8)

    # Vectorized atomics
    if cc >= 90:
        features.add(TargetFeature.VECTORIZED_ATOMICS)

    # Independent epilogue pipelining
    if cc >= 90:
        features.add(TargetFeature.INDEPENDENT_EPILOGUE_PIPELINING)

    # PTX instructions
    if cc >= 80:
        features.add(TargetFeature.PTX_MIN_MAX_NAN)
    # PTX cvt.e2m1 for MXFP4: sm100-sm119 only (NOT sm120+)
    if cc >= 100 and cc < 120:
        features.add(TargetFeature.PTX_CVT_E2M1)

    return features


def _get_amd_features(arch: str) -> set:
    """
    Internal function to determine AMD features from architecture string.
    This mirrors the C++ implementation in TargetArchitecture.cpp.
    """
    features = set()

    if arch == "gfx908":
        # MI100 features
        features.add(TargetFeature.V_DOT_INSTRUCTION)
    elif arch == "gfx90a":
        # MI200 series features
        features.add(TargetFeature.V_DOT_INSTRUCTION)
        features.add(TargetFeature.DIRECT_TO_LDS_SCATTER)
    elif arch in ("gfx940", "gfx942"):
        # MI300 series features
        features.add(TargetFeature.V_DOT_INSTRUCTION)
        features.add(TargetFeature.DIRECT_TO_LDS_SCATTER)
        features.add(TargetFeature.LDS_TRANS_LOAD)

    return features


@constexpr_function
def has_feature(feature: TargetFeature) -> bool:
    """
    Check if the current target supports a specific hardware feature.

    This function replaces compute capability comparisons with explicit feature
    checks, correctly handling cases where newer hardware may lack features
    present in older generations (e.g., sm120 vs sm100).

    Args:
        feature: The hardware feature to check for

    Returns:
        True if the current target supports the feature, False otherwise

    Example:
        >>> if has_feature(TargetFeature.TMA_GATHER):
        ...     use_tma_gather()  # Only runs on sm100-sm119
        >>> if has_feature(TargetFeature.MMA_V5):
        ...     use_mma_v5()  # Only runs on sm100-sm119, NOT sm120
    """
    try:
        active_driver = driver.active
    except RuntimeError:
        # If there is no active driver, return False
        return False

    target = active_driver.get_current_target()
    if target is None:
        return False

    # Get features based on backend
    if target.backend == "cuda":
        if not isinstance(target.arch, int):
            return False
        features = _get_nvidia_features(target.arch)
    elif target.backend == "hip":
        if not isinstance(target.arch, str):
            return False
        features = _get_amd_features(target.arch)
    else:
        return False

    return feature in features


@constexpr_function
def get_best_mma_version() -> int:
    """
    Get the best (highest) MMA version supported by the current target.

    Returns:
        The highest MMA version: 0 (none), 1, 2, 3, or 5

    Example:
        >>> mma_version = get_best_mma_version()
        >>> if mma_version >= 2:
        ...     use_mma_v2_or_better()
    """
    # Check in descending order of preference
    if has_feature(TargetFeature.MMA_V5):
        return 5
    if has_feature(TargetFeature.MMA_V3):
        return 3
    if has_feature(TargetFeature.MMA_V2):
        return 2
    if has_feature(TargetFeature.MMA_V1):
        return 1
    return 0
