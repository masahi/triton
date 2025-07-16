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
    return producedValues;
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

Value mkConstant(PartitionBuilder &builder, StageCluster stageCluster,
                 int value, int width, Partition *partition,
                 WarpSchedule &schedule) {
  assert(partition);
  auto constValue = builder.createInto<arith::ConstantIntOp>(
      *partition, stageCluster, value, width);
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
      mkConstant(builder, stageCluster, 0, 32, producerPartition, schedule));
  schedule.insert(producerPartition, putEnterOp);
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
      mkConstant(builder, stageCluster, 0, 32, producerPartition, schedule),
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

void createArefGet(PartitionBuilder &builder, ArefCreateOp aref,
                   std::string arefTag, ProducedValueInfo producedValue,
                   SetVector<Operation *> users, Partition *consumerPartition,
                   WarpSchedule &schedule) {
  OpBuilder::InsertionGuard g(builder);
  auto loc = producedValue.result.getLoc();
  auto arefBufType = cast<MemDescType>(aref.getOperand(0).getType());

  Value result = producedValue.result;
  StageCluster stageCluster = getStageCluster(result.getDefiningOp());

  SmallVector<Type> buffers{getDataMemDescType(arefBufType, false)};
  auto getEnterOp = builder.createInto<ArefGetEnterOp>(
      *consumerPartition, stageCluster, buffers, aref,
      mkConstant(builder, stageCluster, 0, 32, consumerPartition, schedule));
  Value dataBuf = getEnterOp.getResults()[0];
  schedule.insert(consumerPartition, getEnterOp);
  getEnterOp->setAttr("aref_tag", builder.getStringAttr(arefTag));

  auto createExit = [&](AsyncOp consumerKind) {
    SmallVector<Attribute> consumerAttr{
        AsyncOpAttr::get(aref.getContext(), consumerKind)};
    auto consumersAttr = builder.getArrayAttr(consumerAttr);
    auto getExitOp = builder.createInto<ArefGetExitOp>(
        *consumerPartition, stageCluster, aref,
        mkConstant(builder, stageCluster, 0, 32, consumerPartition, schedule),
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
    auto kind = getConsumerKind(consumers);
    createExit(kind);
    if (auto mmav5 = dyn_cast<MMAv5OpInterface>(consumers.front())) {
      mmav5.setIsAsync(true);
    }
  } else if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
    auto localLoadOp = builder.create<LocalLoadOp>(loc, tensorType, dataBuf);
    newOperand = localLoadOp.getResult();
    schedule.insert(consumerPartition, localLoadOp);
    createExit(AsyncOp::NONE);
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

  SetVector<Operation *> users(result.getUsers().begin(),
                               result.getUsers().end());

  ArefCreateOp aref;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(loop);
    aref = createAref(builder, producedValue);
  }

  auto tag = std::string("aref_") + std::to_string(arefTag);
  auto staleOps = createArefPut(builder, aref, tag, producedValue,
                                producerPartition, schedule);
  createArefGet(builder, aref, tag, producedValue, users, consumerPartition,
                schedule);

  for (auto op : staleOps) {
    op->erase();
  }

  return true;
}

void runArefInsertionOnLoop(scf::ForOp loop, WarpSchedule &schedule) {
  SmallVector<Operation *> opsToArefy;
  loop.walk([&](Operation *op) {
    if (isa<ArefCreateOp, TMEMAllocOp, ArefPutEnterOp, ArefGetEnterOp,
            TMEMLoadOp, TMEMStoreOp, ArefPutExitOp, ArefGetExitOp, scf::YieldOp,
            triton::FuncOp, triton::ReturnOp, scf::ForOp, scf::IfOp>(op))
      return;

    opsToArefy.push_back(op);
  });

  int arefTag = 0;
  auto body = loop.getBody();

  for (auto op : opsToArefy) {
    // otherwise we need to place put/get
    auto producedValues = getProducedValues(op, loop.getBody(), schedule);
    for (auto producedValue : producedValues) {
      PartitionBuilder builder(op->getLoc(), op);
      builder.setInsertionPointAfter(op);
      if (insertArefs(builder, loop, schedule, producedValue, arefTag))
        arefTag++;
    }
  }
}

template <typename EnterOp, typename ExitOp>
ExitOp createCombinedArefOps(SmallVector<EnterOp> &enterOps,
                             SmallVector<ExitOp> &exitOps, ArefCreateOp aref,
                             PartitionBuilder &builder, WarpSchedule &schedule,
                             Operation *enterInsertPoint = nullptr) {
  auto firstEnter = *llvm::min_element(enterOps, [](auto a, auto b) {
    assert(a->getBlock() == b->getBlock());
    return a->isBeforeInBlock(b);
  });

  auto lastExit = *llvm::max_element(exitOps, [](auto a, auto b) {
    assert(a->getBlock() == b->getBlock());
    return a->isBeforeInBlock(b);
  });

  SmallVector<Type> arefEnterBuffers;
  for (auto enterOp : enterOps) {
    arefEnterBuffers.push_back(enterOp.getResult(0).getType());
  }

  llvm::SmallSetVector<Attribute, 5> opAttrsSet;
  for (Operation *exitOp : exitOps) {
    // TODO: an interface for exit op
    if (auto putExit = dyn_cast<ArefPutExitOp>(exitOp)) {
      opAttrsSet.insert(putExit.getAsyncOps()[0]);
    } else if (auto getExit = dyn_cast<ArefGetExitOp>(exitOp)) {
      opAttrsSet.insert(getExit.getAsyncOps()[0]);
    }
  }
  llvm::SmallVector<Attribute> producersOrConsumers(opAttrsSet.begin(),
                                                    opAttrsSet.end());

  if (enterInsertPoint) {
    // Combine get enter need to be placed after combined put exit
    builder.setInsertionPoint(enterInsertPoint);
  } else {
    builder.setInsertionPoint(firstEnter);
  }
  StageCluster stageCluster = getStageCluster(firstEnter);
  auto partition = schedule.getPartition(firstEnter);
  auto zero = mkConstant(builder, stageCluster, 0, 32, partition, schedule);
  auto enter = builder.createInto<EnterOp>(*partition, stageCluster,
                                           arefEnterBuffers, aref, zero);
  builder.setInsertionPoint(lastExit);
  auto exit =
      builder.createInto<ExitOp>(*partition, stageCluster, aref, zero,
                                 builder.getArrayAttr(producersOrConsumers));

  for (auto [idx, enterOp] : llvm::enumerate(enterOps))
    enterOp.getResult(0).replaceAllUsesWith(enter.getResult(idx));

  for (auto op : SmallVector<Operation *>{enter, exit}) {
    op->setAttr("aref_tag", firstEnter->getAttr("aref_tag"));
    schedule.insert(partition, op);
  }

  return lastExit;
}

void combineArefs(scf::ForOp loop, WarpSchedule &schedule) {
  // this subpass will combine arefs into a single one if aref are
  // used by the same op in the same block:

  // %buf_a = alloc(); %aref_a = aref_create %buf_a
  // %buf_b = alloc(); %aref_b = aref_create %buf_b

  // %a = aref_get.enter %aref_a
  // %b = aref_get.enter %aref_b
  //   .. = op .. %a, .. , %b ..
  // aref_get.exit %aref_a
  // aref_get.exit %aref_B

  // %a = aref_put.neter %aref_a
  //  store .. %a
  // aref_put.exit %aref_a
  // %b = aref_put.neter %aref_b
  //  store .. %b
  // aref_put.exit %aref_b

  // becomes

  // %buf_a = alloc(); %buf_b = alloc(); %aref_ab = aref_create %buf_a,
  // %buf_b

  // %a = aref_get.enter %aref_ab
  // %b = aref_get.enter %aref_ab
  //   .. = op .. %a, .. , %b ..
  // aref_get.exit %aref_ab

  // %a,5b = aref_put.enter %aref_ab
  //  store .. %a
  //  store .. %b
  // aref_put.exit %aref_ab

  // for now this happens at MMA sites, so we just visit MMA ops, generic
  // algorithm can be implemented at a later time

  std::function<ArefCreateOp(Value)> findAref =
      [&](Value opnd) -> ArefCreateOp {
    if (!opnd)
      return {};
    if (auto op = opnd.getDefiningOp()) {
      if (isa<WarpGroupDotOp, MMAv5OpInterface>(op)) {
        // TODO, more proper check
        return {};
      } else if (auto enterOp = dyn_cast<ArefGetEnterOp>(op)) {
        return cast<ArefCreateOp>(enterOp.getAref().getDefiningOp());
      } else {
        for (auto operand : op->getOperands())
          if (auto aref = findAref(operand))
            return aref;
      }
    }
    return {};
  };

  SmallVector<SmallVector<ArefCreateOp>> arefsToFuse;
  loop.walk([&](Operation *op) {
    Value Aopnd, Bopnd, AScaleOpnd, BScaleOpnd;
    if (auto wgmma = dyn_cast<WarpGroupDotOp>(op)) {
      Aopnd = wgmma.getA();
      Bopnd = wgmma.getB();
    } else if (auto mmav5 = dyn_cast<TCGen5MMAOp>(op)) {
      Aopnd = mmav5.getA();
      Bopnd = mmav5.getB();
    } else if (auto mmav5scaled = dyn_cast<TCGen5MMAScaledOp>(op)) {
      Aopnd = mmav5scaled.getA();
      Bopnd = mmav5scaled.getB();
      AScaleOpnd = mmav5scaled.getAScale();
      BScaleOpnd = mmav5scaled.getBScale();
    } else {
      return WalkResult::advance();
    }

    auto usedInTheSameBlock = [](ArefCreateOp arefA, ArefCreateOp arefB) {
      Block *putABlock, *getABlock;
      Block *putBBlock, *getBBlock;
      for (auto user : arefA->getUsers()) {
        if (isa<ArefPutEnterOp>(user)) {
          putABlock = user->getBlock();
        } else if (isa<ArefGetEnterOp>(user)) {
          getABlock = user->getBlock();
        }
      }

      for (auto user : arefB->getUsers()) {
        if (isa<ArefPutEnterOp>(user)) {
          putBBlock = user->getBlock();
        } else if (isa<ArefGetEnterOp>(user)) {
          getBBlock = user->getBlock();
        }
      }

      return putABlock == putBBlock && getABlock == getBBlock;
    };

    auto usedOnce = [](ArefCreateOp aref) {
      return llvm::count_if(aref->getUsers(), [](auto) { return true; }) == 4;
    };

    // verify that all put/gets are in the same BB
    auto arefA = findAref(Aopnd);
    auto arefB = findAref(Bopnd);
    SmallVector<ArefCreateOp> arefs;
    if (arefA && arefB && usedOnce(arefA) && usedOnce(arefB) &&
        usedInTheSameBlock(arefA, arefB)) {
      arefs.push_back(arefA);
      arefs.push_back(arefB);
    } else {
      return WalkResult::advance();
    }

    auto arefAScale = findAref(AScaleOpnd);
    if (arefAScale &&
        isa<SharedMemorySpaceAttr>(
            cast<MemDescType>(AScaleOpnd.getType()).getMemorySpace()) &&
        usedOnce(arefAScale) && usedInTheSameBlock(arefAScale, arefA))
      arefs.push_back(arefAScale);

    auto arefBScale = findAref(BScaleOpnd);
    if (arefBScale &&
        isa<SharedMemorySpaceAttr>(
            cast<MemDescType>(BScaleOpnd.getType()).getMemorySpace()) &&
        usedOnce(arefBScale) && usedInTheSameBlock(arefBScale, arefA))
      arefs.push_back(arefBScale);

    if (!arefs.empty())
      arefsToFuse.push_back(arefs);
    return WalkResult::advance();
  });

  //   now fuse arefs
  for (auto arefs : arefsToFuse) {
    SmallVector<ArefPutEnterOp> putEnterOps;
    SmallVector<ArefPutExitOp> putExitOps;
    SmallVector<ArefGetEnterOp> getEnterOps;
    SmallVector<ArefGetExitOp> getExitOps;
    for (auto aref : arefs) {
      for (auto user : aref->getUsers()) {
        if (auto putEnterOp = dyn_cast<ArefPutEnterOp>(user)) {
          putEnterOps.push_back(putEnterOp);
        } else if (auto putExitOp = dyn_cast<ArefPutExitOp>(user)) {
          putExitOps.push_back(putExitOp);
        } else if (auto getEnterOp = dyn_cast<ArefGetEnterOp>(user)) {
          getEnterOps.push_back(getEnterOp);
        } else if (auto getExitOp = dyn_cast<ArefGetExitOp>(user)) {
          getExitOps.push_back(getExitOp);
        }
      }
    }

    // set insertion point at the last aref_create
    SmallVector<Type> arefBufTypes;
    SmallVector<Value> arefBufs;
    for (auto aref : arefs) {
      arefBufTypes.push_back(aref.getOperands()[0].getType());
      arefBufs.push_back(aref.getOperands()[0]);
    }
    auto lastAref = *llvm::max_element(arefs, [](auto a, auto b) {
      assert(a->getBlock() == b->getBlock());
      return a->isBeforeInBlock(b);
    });
    PartitionBuilder builder(lastAref->getLoc(), lastAref);
    auto arefTy =
        ArefType::get(builder.getContext(),
                      TypeArrayAttr::get(builder.getContext(), arefBufTypes));
    auto aref =
        builder.create<ArefCreateOp>(lastAref->getLoc(), arefTy, arefBufs);
    auto lastPutExit =
        createCombinedArefOps(putEnterOps, putExitOps, aref, builder, schedule);
    createCombinedArefOps(getEnterOps, getExitOps, aref, builder, schedule,
                          lastPutExit);

    for (auto putEnterOp : putEnterOps)
      putEnterOp->erase();
    for (auto putExitOp : putExitOps)
      putExitOp->erase();
    for (auto getEnterOp : getEnterOps)
      getEnterOp->erase();
    for (auto getExitOp : getExitOps)
      getExitOp->erase();
    for (auto aref : arefs)
      aref->erase();
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
      combineArefs(loop, *schedule);

      schedule->serialize(loop);
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::triton::createNVWSInsertAref() {
  return std::make_unique<NVWSArefInsertion>();
}
