#ifndef TRITON_TARGET_TARGETARCHITECTURE_H
#define TRITON_TARGET_TARGETARCHITECTURE_H

#include "triton/Target/TargetFeatures.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace triton {

/// NVIDIA GPU architecture families
enum class NVIDIAArch {
  Unknown,
  Volta,             // sm70-sm72
  Turing,            // sm75
  Ampere,            // sm80-sm86
  Ada,               // sm89
  Hopper,            // sm90
  BlackwellUltra,    // sm100, sm103 - full feature set
  BlackwellConsumer, // sm120 - reduced feature set (no MMAv5, no TMA scatter)
};

/// AMD GPU architecture families
enum class AMDArch {
  Unknown,
  GFX908,   // MI100
  GFX90A,   // MI200 series
  GFX940,   // MI300A
  GFX942,   // MI300X
};

/// Backend type for the target
enum class TargetBackend {
  Unknown,
  NVIDIA,
  AMD,
};

/// Represents a specific target architecture and provides access to its
/// feature set. This is the main entry point for querying hardware capabilities.
class TargetArchitecture {
public:
  /// Create a TargetArchitecture for an NVIDIA GPU by compute capability
  /// @param computeCapability The compute capability (e.g., 75, 89, 100, 120)
  /// @return TargetArchitecture with appropriate feature set
  static TargetArchitecture fromNVIDIA(int computeCapability);

  /// Create a TargetArchitecture for an AMD GPU by architecture string
  /// @param archString The architecture string (e.g., "gfx908", "gfx90a")
  /// @return TargetArchitecture with appropriate feature set
  static TargetArchitecture fromAMD(llvm::StringRef archString);

  /// Default constructor creates an unknown/empty target
  TargetArchitecture();

  /// Get the feature set for this architecture
  const TargetFeatureSet& getFeatures() const { return features_; }

  /// Get mutable feature set (for testing/customization)
  TargetFeatureSet& getMutableFeatures() { return features_; }

  /// Get the backend type
  TargetBackend getBackend() const { return backend_; }

  /// Get NVIDIA compute capability (returns 0 if not NVIDIA)
  int getNVIDIAComputeCapability() const;

  /// Get NVIDIA architecture family
  NVIDIAArch getNVIDIAArch() const { return nvidiaArch_; }

  /// Get AMD architecture family
  AMDArch getAMDArch() const { return amdArch_; }

  /// Check if this is a valid/known target
  bool isValid() const { return backend_ != TargetBackend::Unknown; }

private:
  TargetArchitecture(TargetBackend backend, NVIDIAArch nvidiaArch,
                     AMDArch amdArch, int computeCapability);

  TargetBackend backend_;
  NVIDIAArch nvidiaArch_;
  AMDArch amdArch_;
  int computeCapability_; // For NVIDIA only
  TargetFeatureSet features_;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_TARGET_TARGETARCHITECTURE_H
