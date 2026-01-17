add_mlir_library(TritonTarget
  TargetFeatures.cpp
  TargetArchitecture.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  LLVMSupport
)

add_subdirectory(LLVMIR)
