#include "Utilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
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
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
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
  Partition *partition;
  Value result; // result being produced
};

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

SmallVector<ProducedValueInfo> getProducedValues(Operation *op, Block *loopBody,
                                                 WarpSchedule &schedule) {
  SmallVector<ProducedValueInfo> producedValues;
  auto partition = schedule.getPartition(loopBody->findAncestorOpInBlock(*op));
  if (partition == schedule.getRootPartition()) {
    return {};
  }
  for (auto result : op->getResults()) {
    // TODO: Other producer ops
    if (isa<triton::DescriptorOpInterface>(result.getDefiningOp()) ||
        isDescLoadAndAlloc(result)) {
      producedValues.push_back({partition, result});
    }
  }
  return producedValues;
};

ArefCreateOp createAref(OpBuilder &builder, ProducedValueInfo &producedValue) {
  MemDescType arefBufType;

  auto result = producedValue.result;
  assert(!isa<AsyncTokenType>(result.getType()));

  if (auto memDescType = dyn_cast<MemDescType>(result.getType())) {
    arefBufType = getArefbufMemDescType(memDescType, 1);
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    // if result is a value, create memdesctype for location where value will
    // be stored
    MemDescType memDescType;
    Attribute SharedMemorySpace =
        SharedMemorySpaceAttr::get(tensorType.getContext());
    if (auto load = result.getDefiningOp<triton::DescriptorOpInterface>()) {
      auto encoding =
          getEncodingFromDescriptor(load, tensorType, load.getDesc());
      memDescType =
          MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                           encoding, SharedMemorySpace);
    } else {
      for (auto user : producedValue.result.getUsers()) {
        // if user is localAlloc/localStore, uses their memDescType
        if (auto localAlloc = dyn_cast<LocalAllocOp>(user)) {
          memDescType = cast<MemDescType>(localAlloc.getResult().getType());
          break;
        } else if (auto localStore = dyn_cast<LocalStoreOp>(user)) {
          memDescType = cast<MemDescType>(localStore.getDst().getType());
          break;
        }
      }
    }
    if (!memDescType) {
      // The right smem encoding cannot be inferred from the IR. This can
      // happen, for example, in an attention kernel where smem is used to
      // communicate between different warp groups. We need to pick a new
      // encoding for such cases. For now, use a non-swizzled layout.
      // TODO: Use a swizzled one when possible
      auto CTALayout = getCTALayout(tensorType.getEncoding());
      auto newOrder = getOrderForMemory(tensorType);
      auto encoding = SwizzledSharedEncodingAttr::get(
          tensorType.getContext(), 1, 1, 1, newOrder, CTALayout);
      memDescType =
          MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                           encoding, SharedMemorySpace);
    }
    arefBufType = getArefbufMemDescType(memDescType, 1);
  } else {
    // need to support scalar types, similarly  to ranked tensor types
    llvm_unreachable("unsupported type");
  }

  auto loc = producedValue.result.getLoc();

  auto arefTy =
      ArefType::get(builder.getContext(),
                    TypeArrayAttr::get(builder.getContext(), arefBufType));
  assert((isa<SharedMemorySpaceAttr>(arefBufType.getMemorySpace())));
  auto alloc = triton::nvws::createAlloc(builder, loc, arefBufType, Value());
  alloc->setAttr("aref_buffer", builder.getUnitAttr());
  return builder.create<ArefCreateOp>(loc, arefTy, alloc->getResult(0));
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
    newDescLoad->setAttrs(descLoad->getAttrs());
    schedule.insert(producerPartition, newDescLoad);
  } else if (auto descGather =
                 dyn_cast<triton::DescriptorGatherOp>(ttDescLoadOp)) {
    auto newDescGather = builder.create<triton::nvws::DescriptorGatherOp>(
        loc, descGather.getDesc(), descGather.getXOffsets(),
        descGather.getYOffset(), txCount, dataBuf);
    newDescGather->setAttrs(descGather->getAttrs());
    schedule.insert(producerPartition, newDescGather);
  } else {
    llvm_unreachable("unknown descriptor op.");
  }
}

Value mkConstant(OpBuilder &builder, Location loc, int value, int width,
                 Partition *partition, WarpSchedule &schedule) {
  assert(partition);
  auto constValue = builder.create<arith::ConstantIntOp>(loc, value, width);
  schedule.insert(partition, constValue);
  return constValue;
}

StageCluster getStageClusterForProducer(Value producedValue) {
  if (isDescLoadAndAlloc(producedValue) ||
      isGlobalLoadAndAlloc(producedValue)) {
    auto alloc = producedValue.getDefiningOp<LocalAllocOp>();
    auto loadOp = alloc.getSrc().getDefiningOp();
    return getStageCluster(loadOp);
  }
  return getStageCluster(producedValue.getDefiningOp());
}

SmallVector<Operation *> createArefPut(PartitionBuilder &builder,
                                       ArefCreateOp aref, std::string arefTag,
                                       ProducedValueInfo producedValue,
                                       Partition *producerPartition,
                                       WarpSchedule &schedule) {
  auto loc = producedValue.result.getLoc();
  auto arefBufType = cast<MemDescType>(aref.getOperand(0).getType());
  Value result = producedValue.result;
  auto dataBufType = getDataMemDescType(arefBufType, true);
  StageCluster stageCluster = getStageClusterForProducer(result);

  SmallVector<Type> buffers{dataBufType};

  auto putEnterOp = builder.createInto<ArefPutEnterOp>(
      *producerPartition, stageCluster, buffers, aref,
      mkConstant(builder, loc, 0, 32, producerPartition, schedule));
  schedule.insert(producerPartition, putEnterOp);
  // TODO: is this necessary
  putEnterOp->setAttr("aref_tag", builder.getStringAttr(arefTag));
  auto dataBuf = putEnterOp.getResults()[0];

  auto producerKind = AsyncOp::NONE;
  SmallVector<Operation *> staleOps;
  if (isDescLoadAndAlloc(result)) {
    auto alloc = result.getDefiningOp<LocalAllocOp>();
    auto descOp = alloc.getSrc().getDefiningOp();
    createNVWSDescriptorLoadOp(builder, descOp, dataBuf, producerPartition,
                               schedule, loc);
    producerKind = AsyncOp::TMALoad;
    staleOps.push_back(alloc);
    staleOps.push_back(descOp);
  } else if (isGlobalLoadAndAlloc(result)) {
    llvm_unreachable("cpasync not supported yet");
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    auto op = result.getDefiningOp();
    if (op && isa<triton::DescriptorOpInterface>(op)) {
      createNVWSDescriptorLoadOp(builder, op, dataBuf, producerPartition,
                                 schedule, loc);
      producerKind = AsyncOp::TMALoad;
      staleOps.push_back(op);
    } else if (op && isa<triton::LoadOp>(op)) {
      llvm_unreachable("cpasync not supported yet");
    } else {
      auto storeOp = builder.create<LocalStoreOp>(loc, result, dataBuf);
      schedule.insert(producerPartition, storeOp);
    }
  } else {
    llvm_unreachable("unsupported type");
  }

  auto putExitOp = builder.createInto<ArefPutExitOp>(
      *producerPartition, stageCluster, aref,
      mkConstant(builder, loc, 0, 32, producerPartition, schedule),
      builder.getArrayAttr(SmallVector<Attribute>{
          AsyncOpAttr::get(aref.getContext(), producerKind)}));
  putExitOp->setAttr("aref_tag", builder.getStringAttr(arefTag));
  schedule.insert(producerPartition, putExitOp);
  return staleOps;
};

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

SmallVector<Attribute>
getConsumerAsyncOpKinds(const SetVector<Operation *> &consumers,
                        MLIRContext *ctx) {
  SetVector<AsyncOp> kindSet;
  for (auto consumer : consumers) {
    if (isa<WarpGroupDotOp>(consumer)) {
      kindSet.insert(AsyncOp::WGMMA);
    } else if (auto mmav5 = dyn_cast<MMAv5OpInterface>(consumer)) {
      kindSet.insert(AsyncOp::TC5MMA);
    } else {
      kindSet.insert(AsyncOp::NONE);
    }
  }

  SmallVector<Attribute> kindAttrs;
  for (auto kind : kindSet) {
    kindAttrs.push_back(AsyncOpAttr::get(ctx, kind));
  }

  return kindAttrs;
}

void createArefGet(PartitionBuilder &builder, ArefCreateOp aref,
                   std::string arefTag, ProducedValueInfo producedValue,
                   Partition *consumerPartition, WarpSchedule &schedule,
                   PostDominanceInfo &postDomInfo) {
  OpBuilder::InsertionGuard g(builder);
  auto loc = producedValue.result.getLoc();
  auto arefBufType = cast<MemDescType>(aref.getOperand(0).getType());

  Value result = producedValue.result;
  StageCluster stageCluster = getStageCluster(result.getDefiningOp());

  SmallVector<Type> buffers{getDataMemDescType(arefBufType, false)};
  auto getEnterOp = builder.createInto<ArefGetEnterOp>(
      *consumerPartition, stageCluster, buffers, aref,
      mkConstant(builder, loc, 0, 32, consumerPartition, schedule));
  Value dataBuf = getEnterOp.getResults()[0];
  schedule.insert(consumerPartition, getEnterOp);
  getEnterOp->setAttr("aref_tag", builder.getStringAttr(arefTag));

  auto consumers = getTransitiveConsumers(result.getDefiningOp());
  auto asyncKinds = getConsumerAsyncOpKinds(consumers, aref.getContext());

  if (auto memDescType = dyn_cast<MemDescType>(result.getType())) {
    auto insertPoint = *llvm::min_element(consumers, [&](auto &lhs, auto &rhs) {
      return postDomInfo.postDominates(lhs, rhs);
    });
    builder.setInsertionPointAfter(insertPoint);
    result.replaceAllUsesWith(dataBuf);
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    auto localLoadOp = builder.create<LocalLoadOp>(loc, tensorType, dataBuf);
    result.replaceAllUsesWith(localLoadOp.getResult());
    schedule.insert(consumerPartition, localLoadOp);
  } else {
    llvm_unreachable("unsupported type");
  }

  auto getExitOp = builder.createInto<ArefGetExitOp>(
      *consumerPartition, stageCluster, aref,
      mkConstant(builder, loc, 0, 32, consumerPartition, schedule),
      builder.getArrayAttr(asyncKinds));

  getExitOp->setAttr("aref_tag", builder.getStringAttr(arefTag));
  schedule.insert(consumerPartition, getExitOp);

  for (auto consumer : consumers) {
    if (auto mmav5 = dyn_cast<MMAv5OpInterface>(consumer)) {
      mmav5.setIsAsync(true);
    }
  }
};

bool insertArefs(PartitionBuilder &builder, scf::ForOp loop,
                 WarpSchedule &schedule, ProducedValueInfo producedValue,
                 int arefTag) {
  Partition *consumerPartition = nullptr;
  auto [producerPartition, result] = producedValue;
  assert(producerPartition);
  for (auto &useOpnd : result.getUses()) {
    Partition *userPartition = schedule.getPartition((useOpnd.getOwner()));
    if (producerPartition != userPartition) {
      consumerPartition = userPartition;
      break;
    }
  }

  if (!consumerPartition) {
    return false;
  }
  // we also enforce that there is at least one user of the result
  assert(llvm::count_if(result.getUsers(), [](auto) { return true; }) >= 1);

  ArefCreateOp aref;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(loop);
    aref = createAref(builder, producedValue);
  }

  auto tag = std::string("aref_") + std::to_string(arefTag);
  auto staleOps = createArefPut(builder, aref, tag, producedValue,
                                producerPartition, schedule);

  PostDominanceInfo postDomInfo(loop);
  createArefGet(builder, aref, tag, producedValue, consumerPartition, schedule,
                postDomInfo);

  for (auto op : staleOps) {
    op->erase();
  }

  return true;
}

void runArefInsertionOnLoop(scf::ForOp loop, WarpSchedule &schedule) {
  SmallVector<Operation *> ops;
  loop.walk([&](Operation *op) {
    if (isa<ArefCreateOp, TMEMAllocOp, ArefPutEnterOp, ArefGetEnterOp,
            TMEMLoadOp, TMEMStoreOp, ArefPutExitOp, ArefGetExitOp, scf::YieldOp,
            triton::FuncOp, triton::ReturnOp, scf::ForOp, scf::IfOp>(op))
      return;

    ops.push_back(op);
  });

  int arefTag = 0;

  for (auto op : ops) {
    auto producedValues = getProducedValues(op, loop.getBody(), schedule);
    for (auto producedValue : producedValues) {
      PartitionBuilder builder(op->getLoc(), op);
      builder.setInsertionPointAfter(op);
      if (insertArefs(builder, loop, schedule, producedValue, arefTag))
        arefTag++;
    }
  }
}

class NVWSArefInsertion : public NVWSInsertArefBase<NVWSArefInsertion> {
public:
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) {
      if (loop->hasAttr(triton::kWarpSpecializeAttrName))
        loops.push_back(loop);
    });
    for (scf::ForOp loop : loops) {
      FailureOr<WarpSchedule> schedule = WarpSchedule::deserialize(loop);
      if (failed(schedule))
        continue;

      runArefInsertionOnLoop(loop, *schedule);

      schedule->serialize(loop);
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::triton::createNVWSInsertAref() {
  return std::make_unique<NVWSArefInsertion>();
}
