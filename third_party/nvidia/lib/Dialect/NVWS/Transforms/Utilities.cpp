#include "Utilities.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace nvws {

MemDescType getDataMemDescType(MemDescType memDescType, bool mutableMemory) {
  auto shape = memDescType.getShape();
  SmallVector<int64_t> dataShape(shape.begin() + 1, shape.end());
  return MemDescType::get(dataShape, memDescType.getElementType(),
                          memDescType.getEncoding(),
                          memDescType.getMemorySpace(), mutableMemory);
};

Operation *createAlloc(OpBuilder &builder, Location loc,
                       MemDescType memDescType, Value src) {
  if (isa<SharedMemorySpaceAttr>(memDescType.getMemorySpace()))
    return builder.create<LocalAllocOp>(loc, memDescType, src);
  else {
    assert(isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace()));
    return builder.create<TMEMAllocOp>(loc, memDescType, src);
  }
}

MemDescType getArefbufMemDescType(MemDescType memDescType, int32_t AREF_SIZE) {
  auto shape = memDescType.getShape();
  SmallVector<int64_t> bufferShape(shape.begin(), shape.end());
  bufferShape.insert(bufferShape.begin(), AREF_SIZE);
  return MemDescType::get(bufferShape, memDescType.getElementType(),
                          memDescType.getEncoding(),
                          memDescType.getMemorySpace(), true);
}

} // namespace nvws
} // namespace triton
} // namespace mlir
