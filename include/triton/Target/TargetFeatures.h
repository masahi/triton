#ifndef TRITON_TARGET_TARGETFEATURES_H
#define TRITON_TARGET_TARGETFEATURES_H

#include "llvm/ADT/SmallBitVector.h"
#include <cstddef>

namespace mlir {
namespace triton {

/// Enumeration of hardware features that can be queried across different
/// target architectures. This replaces numeric compute capability comparisons
/// with explicit feature checks.
enum class TargetFeature {
  // Memory operations
  LdMatrix,         // Load matrix from shared memory (sm75+)
  StMatrix,         // Store matrix to shared memory (sm90+)
  LdStMatrixB8,     // 8-bit ldstmatrix (sm100-sm119 only, NOT sm120)

  // TMA (Tensor Memory Accelerator)
  TMAGeneral,       // Basic TMA support (sm90+)
  TMAGather,        // TMA gather operations (sm100-sm119 only, NOT sm120)
  TMAScatter,       // TMA scatter operations (sm100-sm119 only, NOT sm120)

  // MMA (Matrix Multiply-Accumulate) versions
  MMAv1,            // MMA version 1 (sm70+)
  MMAv2,            // MMA version 2 (sm75+, sm120+)
  MMAv3,            // MMA version 3 (sm90-sm99 only)
  MMAv5,            // MMA version 5 (sm100-sm119 only, NOT sm120)
  MMAv2FP8Native,   // Native FP8 in MMAv2 (sm89, sm120 only)

  // Numeric types
  NativeMXFP4,      // Native MXFP4 support (sm100-sm119 only, NOT sm120)
  NativeFP8,        // Native FP8 support (sm89+)

  // Execution features
  VectorizedAtomics,              // Vectorized atomic operations
  IndependentEpiloguePipelining,  // Independent epilogue pipelining (sm90+)

  // PTX instructions
  PTXMinMaxNaN,     // PTX min/max with NaN handling
  PTXCvtE2M1,       // PTX cvt.e2m1 instruction (MXFP4)

  // AMD-specific features
  VDotInstruction,        // v_dot instruction
  DirectToLDSScatter,     // Direct to LDS scatter
  LDSTransLoad,           // LDS transposed load

  // Sentinel value for sizing bit vectors
  NumFeatures
};

/// A set of target features supported by a specific architecture.
/// Provides efficient storage and query operations for feature flags.
class TargetFeatureSet {
public:
  TargetFeatureSet();

  /// Check if a specific feature is supported
  bool has(TargetFeature feature) const;

  /// Add a feature to the set
  void add(TargetFeature feature);

  /// Remove a feature from the set
  void remove(TargetFeature feature);

  /// Get the best (highest) MMA version supported.
  /// Returns: 0 (none), 1, 2, 3, or 5
  int getBestMMAVersion() const;

  /// Get the shared memory capacity in bytes
  size_t getSharedMemoryCapacity() const;

  /// Set the shared memory capacity in bytes
  void setSharedMemoryCapacity(size_t capacity);

  /// Check if any features are set
  bool empty() const;

  /// Clear all features
  void clear();

private:
  llvm::SmallBitVector features_;
  size_t sharedMemoryCapacity_ = 0;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_TARGET_TARGETFEATURES_H
