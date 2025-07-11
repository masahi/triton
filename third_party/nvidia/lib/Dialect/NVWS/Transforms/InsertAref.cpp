#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#define GEN_PASS_CLASSES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-aref-insertion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;
using namespace triton::nvws;

struct ProducedValueInfo {
  const Partition *partition;
  Value result; // result being produced
};

Operation *createAlloc(OpBuilder &builder, Location loc,
                       MemDescType memDescType, Value src) {
  if (isa<SharedMemorySpaceAttr>(memDescType.getMemorySpace()))
    return builder.create<LocalAllocOp>(loc, memDescType, src);
  else {
    assert(isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(
        memDescType.getMemorySpace()));
    return builder.create<triton::nvidia_gpu::TMEMAllocOp>(loc, memDescType,
                                                           src);
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

SmallVector<ProducedValueInfo> getProducedValues(Operation *op,
                                                 const WarpSchedule &schedule) {
  auto partition = schedule.getPartition(op);
  SmallVector<ProducedValueInfo> producedValues;
  for (auto result : op->getResults()) {
    producedValues.push_back({partition, result});
  }
  return producedValues;
};

ArefCreateOp createAref(OpBuilder &builder, ProducedValueInfo &producedValue) {
  MemDescType arefBufType;

  auto result = producedValue.result;
  assert(!isa<AsyncTokenType>(result.getType()));

  if (auto memDescType = dyn_cast<MemDescType>(result.getType())) {
    arefBufType = getArefbufMemDescType(memDescType, 1);
    auto loc = producedValue.result.getLoc();

    auto arefTy =
        ArefType::get(builder.getContext(),
                      TypeArrayAttr::get(builder.getContext(), arefBufType));
    assert((isa<SharedMemorySpaceAttr>(arefBufType.getMemorySpace())));
    auto alloc = createAlloc(builder, loc, arefBufType, Value());
    alloc->setAttr("aref_buffer", builder.getUnitAttr());
    return builder.create<ArefCreateOp>(loc, arefTy, alloc->getResult(0));
    ;
  }

  return nullptr;
}

bool isDescLoadAndAlloc(Value result) {
  auto alloc = result.getDefiningOp<LocalAllocOp>();
  if (!alloc)
    return false;
  return alloc.getSrc().getDefiningOp<triton::DescriptorOpInterface>() !=
         nullptr;
}

bool isGlobalLoadAndAlloc(Value result) {
  auto alloc = result.getDefiningOp<LocalAllocOp>();
  if (!alloc)
    return false;
  return alloc.getSrc().getDefiningOp<triton::LoadOp>() != nullptr;
}

int getTxCount(Operation *descOp) {
  auto getTensorTypeAndDesc =
      [](Operation *op) -> std::pair<RankedTensorType, Value> {
    if (auto loadOp = dyn_cast<triton::DescriptorLoadOp>(op)) {
      return {loadOp.getType(), loadOp.getDesc()};
    } else if (auto gatherOp = dyn_cast<triton::DescriptorGatherOp>(op)) {
      return {gatherOp.getType(), gatherOp.getDesc()};
    } else {
      llvm_unreachable("Unsupported operation type");
    }
  };
  auto [tensorType, desc] = getTensorTypeAndDesc(descOp);
  auto encoding = getEncodingFromDescriptor(descOp, tensorType, desc);
  auto shapePerCTA = getShapePerCTA(encoding, tensorType.getShape());
  return product(shapePerCTA) *
         tensorType.getElementType().getIntOrFloatBitWidth() / 8;
}

void createNVWSDescriptorLoadOp(OpBuilder &builder, Operation *ttDescLoadOp,
                                Value dataBuf, Partition *producerPartition,
                                WarpSchedule &schedule, Location loc) {
  auto txCount = getTxCount(ttDescLoadOp);
  if (auto descLoad = dyn_cast<triton::DescriptorLoadOp>(ttDescLoadOp)) {
    auto newDescLoad = builder.create<triton::nvws::DescriptorLoadOp>(
        loc, descLoad.getDesc(), descLoad.getIndices(), txCount, dataBuf,
        descLoad.getCache(), descLoad.getEvict());
    schedule.insert(producerPartition, newDescLoad);
  } else if (auto descGather =
                 dyn_cast<triton::DescriptorGatherOp>(ttDescLoadOp)) {
    auto newDescGather = builder.create<triton::nvws::DescriptorGatherOp>(
        loc, descGather.getDesc(), descGather.getXOffsets(),
        descGather.getYOffset(), txCount, dataBuf);
    schedule.insert(producerPartition, newDescGather);
  } else {
    llvm_unreachable("unknown descriptor op.");
  }
}

MemDescType getDataMemDescType(MemDescType memDescType,
                                    bool mutableMemory) {
  auto shape = memDescType.getShape();
  SmallVector<int64_t> dataShape(shape.begin() + 1, shape.end());
  return MemDescType::get(dataShape, memDescType.getElementType(),
                               memDescType.getEncoding(),
                               memDescType.getMemorySpace(), mutableMemory);
};

Value mkConstant(OpBuilder &builder, Location loc, int value, int width,
                 Partition* partition, WarpSchedule& schedule) {
  auto constValue = builder.create<arith::ConstantIntOp>(loc, value, width);
  if (partition) {
    schedule.insert(partition, constValue);
  }

  return constValue;
}

SmallVector<Operation *>
createArefPut(OpBuilder &builder, ArefCreateOp aref, std::string arefTag,
              ProducedValueInfo producedValue, Partition* producerPartition, WarpSchedule& schedule) {
  auto loc = producedValue.result.getLoc();
  auto arefBufType = cast<MemDescType>(aref.getOperand(0).getType());
  Value result = producedValue.result;
  auto dataBufType = getDataMemDescType(arefBufType, true);

  SmallVector<Type> buffers{dataBufType};

  auto putEnterOp = builder.create<ArefPutEnterOp>(
      loc, buffers, aref,
      mkConstant(builder, loc, 0, 32, producerPartition, schedule));
  schedule.insert(producerPartition, putEnterOp);
  putEnterOp->setAttr("aref_tag", builder.getStringAttr(arefTag));
  auto dataBuf = putEnterOp.getResults()[0];

  auto producerKind = AsyncOp::NONE;
  SmallVector<Operation *> staleOps;

  if (isDescLoadAndAlloc(result)) {
    auto alloc = result.getDefiningOp<LocalAllocOp>();
    auto descOp = alloc.getSrc().getDefiningOp();
    createNVWSDescriptorLoadOp(builder, descOp, dataBuf, producerPartition, schedule, loc);
    producerKind = AsyncOp::TMALoad;
    staleOps.push_back(alloc);
    staleOps.push_back(descOp);
  } else if (isGlobalLoadAndAlloc(result)) {
    auto alloc = result.getDefiningOp<LocalAllocOp>();
    auto loadOp = alloc.getSrc().getDefiningOp<triton::LoadOp>();
    assert(loadOp);
    auto newLoad = builder.create<AsyncCopyGlobalToLocalOp>(
        loc, loadOp.getPtr(), dataBuf, loadOp.getMask(), loadOp.getOther(),
        loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    schedule.insert(producerPartition, newLoad);
    producerKind = AsyncOp::CpAsync;
    staleOps.push_back(alloc);
    staleOps.push_back(loadOp);
  } else {
    llvm_unreachable("Aref for value NYT");
  }

  auto putExitOp = builder.create<ArefPutExitOp>(
      loc, aref, mkConstant(builder, loc, 0, 32, producerPartition, schedule),
      builder.getArrayAttr(SmallVector<Attribute>{
          AsyncOpAttr::get(aref.getContext(), producerKind)}));
  putExitOp->setAttr("aref_tag", builder.getStringAttr(arefTag));
  schedule.insert(producerPartition, putExitOp);

  return staleOps;
};

enum class BlockScope {
  UNSUPPORTED,
  SAME_BLOCK,
  NESTED_INSIDE,
};

BlockScope getBlockScope(Block *from, Block *to) {
  if (from == to)
    return BlockScope::SAME_BLOCK;

  auto block = to;
  while (block && block != from)
    block = block->getParentOp()->getBlock();
  if (block == from)
    return BlockScope::NESTED_INSIDE;

  return BlockScope::UNSUPPORTED;
}

SetVector<Operation *> getTransitiveConsumers(Operation *op) {
  SetVector<Operation *> opConsumers;
  for (auto user : op->getUsers()) {
    if (llvm::count_if(user->getResults(), [](auto res) {
          return isa<MemDescType>(res.getType());
        }) == 0) {
      opConsumers.insert(user);
    } else {
      auto consumers = getTransitiveConsumers(user);
      opConsumers.insert(consumers.begin(), consumers.end());
    }
  }
  return opConsumers;
}

AsyncOp getConsumerKind(const SetVector<Operation *> &consumers) {
  assert(!consumers.empty());
  auto consumer = consumers.front();
  if (isa<WarpGroupDotOp>(consumer)) {
    return AsyncOp::WGMMA;
  } else if (auto mmav5 = dyn_cast<MMAv5OpInterface>(consumer)) {
    return AsyncOp::TC5MMA;
  }
  return AsyncOp::NONE;
}

Operation *getExitInsertPoint(Block *producerBlock,
                              const SetVector<Operation *> &consumers) {
  DenseMap<Operation *, int> opOrdering;
  producerBlock->walk(
      [&](Operation *op) { opOrdering[op] = opOrdering.size(); });

  SetVector<Operation *> validConsumers;
  for (auto consumer : consumers) {
    if (getBlockScope(producerBlock, consumer->getBlock()) !=
        BlockScope::UNSUPPORTED)
      validConsumers.insert(consumer);
  }
  assert(!validConsumers.empty());

  auto lastConsumer = *llvm::max_element(validConsumers, [&](auto a, auto b) {
    return opOrdering.at(a) < opOrdering.at(b);
  });

  auto consumerScope = getBlockScope(producerBlock, lastConsumer->getBlock());
  if (BlockScope::SAME_BLOCK == consumerScope) {
    return lastConsumer;
  } else if (BlockScope::NESTED_INSIDE == consumerScope) {
    auto regionOp = lastConsumer->getParentOp();
    while (regionOp->getBlock() != producerBlock) {
      regionOp = regionOp->getParentOp();
    }
    return regionOp;
  } else {
    llvm_unreachable("unsupported consumer scope");
  }
  return nullptr;
}

void createArefGet(OpBuilder &builder, ArefCreateOp aref,
                   std::string arefTag, ProducedValueInfo producedValue,
                   SetVector<Operation *> users, Partition* consumerPartition,
		   WarpSchedule& schedule) {
  OpBuilder::InsertionGuard g(builder);
  auto loc = producedValue.result.getLoc();
  auto arefBufType = cast<MemDescType>(aref.getOperand(0).getType());

  Value result = producedValue.result;

  SmallVector<Type> buffers{getDataMemDescType(arefBufType, false)};
  auto getEnterOp = builder.create<ArefGetEnterOp>(
      loc, buffers, aref,
      mkConstant(builder, loc, 0, 32, consumerPartition, schedule));
  Value dataBuf = getEnterOp.getResults()[0];
  schedule.insert(consumerPartition, getEnterOp);
  getEnterOp->setAttr("aref_tag", builder.getStringAttr(arefTag));

  auto createExit = [&](AsyncOp consumerKind) {
    SmallVector<Attribute> consumerAttr{
        AsyncOpAttr::get(aref.getContext(), consumerKind)};
    auto consumersAttr = builder.getArrayAttr(consumerAttr);
    auto getExitOp = builder.create<ArefGetExitOp>(
        loc, aref, mkConstant(builder, loc, 0, 32, consumerPartition, schedule),
        consumersAttr);
    getExitOp->setAttr("aref_tag", builder.getStringAttr(arefTag));
    schedule.insert(consumerPartition, getExitOp);
  };

  Value newOperand;
  if (auto memDescType = dyn_cast<MemDescType>(result.getType())) {
    newOperand = dataBuf;
    auto consumers = getTransitiveConsumers(result.getDefiningOp());
    auto insertPoint =
        getExitInsertPoint(result.getDefiningOp()->getBlock(), consumers);
    builder.setInsertionPointAfter(insertPoint);
    createExit(getConsumerKind(consumers));
  } else {
    llvm_unreachable("unsupported type");
  }

  // update result operand with newOperand
  for (auto user : users) {
    bool updatedOperand = false;
    for (auto [i, operand] : llvm::enumerate(user->getOperands())) {
      if (result == operand) {
        user->setOperand(i, newOperand);
        updatedOperand = true;
      }
    }
    assert(updatedOperand);
  }
};

class NVWSArefInsertion : public NVWSInsertArefBase<NVWSArefInsertion> {
public:
  void runOnOperation() override {
    // auto mod = getOperation();
    // mod.walk([&](triton::FuncOp funcOp) { runArefInsertionOnFunc(funcOp); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::triton::createNVWSInsertAref() {
  return std::make_unique<NVWSArefInsertion>();
}
