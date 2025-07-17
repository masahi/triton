#ifndef NVIDIA_NVWS_TRANSFORMS_UTILITY_H_
#define NVIDIA_NVWS_TRANSFORMS_UTILITY_H_

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace nvws {

gpu::MemDescType getDataMemDescType(gpu::MemDescType memDescType,
                                    bool mutableMemory);

Operation *createAlloc(OpBuilder &builder, Location loc,
                       gpu::MemDescType memDescType, Value src);

gpu::MemDescType getArefbufMemDescType(gpu::MemDescType memDescType,
                                       int32_t AREF_SIZE);

} // namespace nvws
} // namespace triton
} // namespace mlir

#endif // NVIDIA_NVWS_TRANSFORMS_UTILITY_H_
