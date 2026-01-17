#include "triton/Target/TargetArchitecture.h"

namespace mlir {
namespace triton {

// Helper function to populate NVIDIA feature set based on compute capability
static TargetFeatureSet getNVIDIAFeatures(NVIDIAArch arch, int cc) {
  TargetFeatureSet features;

  // Memory operations
  if (cc >= 75) {
    features.add(TargetFeature::LdMatrix);
  }
  if (cc >= 90) {
    features.add(TargetFeature::StMatrix);
  }
  // LdStMatrixB8: Only sm100-sm119 (NOT sm120+)
  if (cc >= 100 && cc < 120) {
    features.add(TargetFeature::LdStMatrixB8);
  }

  // TMA (Tensor Memory Accelerator)
  if (cc >= 90) {
    features.add(TargetFeature::TMAGeneral);
  }
  // TMA Gather/Scatter: Only sm100-sm119 (NOT sm120+)
  if (cc >= 100 && cc < 120) {
    features.add(TargetFeature::TMAGather);
    features.add(TargetFeature::TMAScatter);
  }

  // MMA versions
  if (cc >= 70) {
    features.add(TargetFeature::MMAv1);
  }
  if (cc >= 75) {
    features.add(TargetFeature::MMAv2);
  }
  // MMAv3: sm90-sm99 only
  if (cc >= 90 && cc < 100) {
    features.add(TargetFeature::MMAv3);
  }
  // MMAv5: sm100-sm119 only (NOT sm120+)
  if (cc >= 100 && cc < 120) {
    features.add(TargetFeature::MMAv5);
  }

  // Special: Native FP8 in MMAv2 for sm89 and sm120
  if (cc == 89 || cc == 120) {
    features.add(TargetFeature::MMAv2FP8Native);
  }

  // Native MXFP4: sm100-sm119 only (NOT sm120+)
  if (cc >= 100 && cc < 120) {
    features.add(TargetFeature::NativeMXFP4);
  }

  // Native FP8 (general)
  if (cc >= 89) {
    features.add(TargetFeature::NativeFP8);
  }

  // Vectorized atomics
  if (cc >= 90) {
    features.add(TargetFeature::VectorizedAtomics);
  }

  // Independent epilogue pipelining
  if (cc >= 90) {
    features.add(TargetFeature::IndependentEpiloguePipelining);
  }

  // PTX instructions
  if (cc >= 80) {
    features.add(TargetFeature::PTXMinMaxNaN);
  }
  // PTX cvt.e2m1 for MXFP4: sm100-sm119 only (NOT sm120+)
  if (cc >= 100 && cc < 120) {
    features.add(TargetFeature::PTXCvtE2M1);
  }

  // Shared memory capacity
  if (cc >= 80) {
    // Ampere and later have larger shared memory
    features.setSharedMemoryCapacity(164 * 1024); // 164 KB
  } else if (cc >= 70) {
    // Volta has 96 KB
    features.setSharedMemoryCapacity(96 * 1024);
  }

  return features;
}

// Helper function to determine NVIDIA architecture family from compute capability
static NVIDIAArch getNVIDIAArchFromCC(int cc) {
  if (cc >= 120 && cc < 130) {
    return NVIDIAArch::BlackwellConsumer;
  } else if (cc >= 100 && cc < 120) {
    return NVIDIAArch::BlackwellUltra;
  } else if (cc >= 90 && cc < 100) {
    return NVIDIAArch::Hopper;
  } else if (cc >= 89 && cc < 90) {
    return NVIDIAArch::Ada;
  } else if (cc >= 80 && cc < 89) {
    return NVIDIAArch::Ampere;
  } else if (cc >= 75 && cc < 80) {
    return NVIDIAArch::Turing;
  } else if (cc >= 70 && cc < 75) {
    return NVIDIAArch::Volta;
  }
  return NVIDIAArch::Unknown;
}

// Helper function to populate AMD feature set
static TargetFeatureSet getAMDFeatures(AMDArch arch) {
  TargetFeatureSet features;

  switch (arch) {
  case AMDArch::GFX908:
    // MI100 features
    features.add(TargetFeature::VDotInstruction);
    features.setSharedMemoryCapacity(64 * 1024); // 64 KB LDS
    break;

  case AMDArch::GFX90A:
    // MI200 series features
    features.add(TargetFeature::VDotInstruction);
    features.add(TargetFeature::DirectToLDSScatter);
    features.setSharedMemoryCapacity(64 * 1024); // 64 KB LDS
    break;

  case AMDArch::GFX940:
  case AMDArch::GFX942:
    // MI300 series features
    features.add(TargetFeature::VDotInstruction);
    features.add(TargetFeature::DirectToLDSScatter);
    features.add(TargetFeature::LDSTransLoad);
    features.setSharedMemoryCapacity(64 * 1024); // 64 KB LDS
    break;

  default:
    break;
  }

  return features;
}

// Helper function to parse AMD architecture string
static AMDArch parseAMDArch(llvm::StringRef archString) {
  if (archString == "gfx908")
    return AMDArch::GFX908;
  if (archString == "gfx90a")
    return AMDArch::GFX90A;
  if (archString == "gfx940")
    return AMDArch::GFX940;
  if (archString == "gfx942")
    return AMDArch::GFX942;
  return AMDArch::Unknown;
}

// TargetArchitecture implementation

TargetArchitecture::TargetArchitecture()
    : backend_(TargetBackend::Unknown), nvidiaArch_(NVIDIAArch::Unknown),
      amdArch_(AMDArch::Unknown), computeCapability_(0) {}

TargetArchitecture::TargetArchitecture(TargetBackend backend,
                                       NVIDIAArch nvidiaArch, AMDArch amdArch,
                                       int computeCapability)
    : backend_(backend), nvidiaArch_(nvidiaArch), amdArch_(amdArch),
      computeCapability_(computeCapability) {
  // Populate features based on backend
  if (backend == TargetBackend::NVIDIA) {
    features_ = getNVIDIAFeatures(nvidiaArch, computeCapability);
  } else if (backend == TargetBackend::AMD) {
    features_ = getAMDFeatures(amdArch);
  }
}

TargetArchitecture TargetArchitecture::fromNVIDIA(int computeCapability) {
  NVIDIAArch arch = getNVIDIAArchFromCC(computeCapability);
  return TargetArchitecture(TargetBackend::NVIDIA, arch, AMDArch::Unknown,
                            computeCapability);
}

TargetArchitecture TargetArchitecture::fromAMD(llvm::StringRef archString) {
  AMDArch arch = parseAMDArch(archString);
  return TargetArchitecture(TargetBackend::AMD, NVIDIAArch::Unknown, arch, 0);
}

int TargetArchitecture::getNVIDIAComputeCapability() const {
  if (backend_ != TargetBackend::NVIDIA)
    return 0;
  return computeCapability_;
}

} // namespace triton
} // namespace mlir
