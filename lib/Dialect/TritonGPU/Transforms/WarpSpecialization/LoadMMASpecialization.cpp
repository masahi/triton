#include "PartitionBuilder.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

//===----------------------------------------------------------------------===//
// getPartitionScheme
//===----------------------------------------------------------------------===//

namespace {

struct PipelinedMMA {
  PipelinedMMA(ttng::MMAv5OpInterface mmaOp) : mmaOp(mmaOp) {}

  ttng::MMAv5OpInterface mmaOp;
};
} // namespace

static SmallVector<PipelinedMMA>
getPartitionScheme(scf::ForOp loop, const WarpSchedule &schedule) {
  SmallVector<PipelinedMMA> mmas;

  for (auto mmaOp : loop.getOps<ttng::MMAv5OpInterface>()) {
    mmas.emplace_back(mmaOp);
  }

  return std::move(mmas);
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static std::pair<Value, Value> postIncrementModulo(ImplicitLocOpBuilder &b,
                                                   Value index, Value phase,
                                                   unsigned numStages) {
  auto intCst = [&](int value) {
    return b.create<arith::ConstantIntOp>(value, 32);
  };
  Value nextIndex = b.create<arith::AddIOp>(index, intCst(1));
  Value nextPhase = b.create<arith::XOrIOp>(phase, intCst(1));

  Value rollover = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, nextIndex,
                                           intCst(numStages));
  nextIndex = b.create<arith::SelectOp>(rollover, intCst(0), nextIndex);
  nextPhase = b.create<arith::SelectOp>(rollover, nextPhase, phase);

  return {nextIndex, nextPhase};
}

static std::pair<BlockArgument, BlockArgument>
addIndexAndPhase(PartitionBuilder &b, scf::ForOp &loop, unsigned numStages,
                 Value epilogue = {}) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);

  // Index and phase both start at 0.
  loop = addIterArgsToLoop(b, loop, {b.intCst(0), b.intCst(0)});
  auto newArgs = loop.getRegionIterArgs().take_back(2);
  BlockArgument index = newArgs[0];
  BlockArgument phase = newArgs[1];

  // Post-increment the index and phase.
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  b.setInsertionPoint(yield);

  auto [nextIndex, nextPhase] = postIncrementModulo(b, index, phase, numStages);
  if (epilogue) {
    nextIndex = b.create<arith::SelectOp>(epilogue, nextIndex, index);
    nextPhase = b.create<arith::SelectOp>(epilogue, nextPhase, phase);
  }
  yield->insertOperands(yield.getNumOperands(), {nextIndex, nextPhase});

  return {index, phase};
}

static Value getUserPrecondition(ImplicitLocOpBuilder &b, scf::ForOp loop,
                                 Operation *domOp) {
  // If the use is inside a loop besides the actual loop being pipelined, we
  // have to hoist the use up to that loop, otherwise the barriers will be
  // inserted in the loop.
  for (Operation *userLoop;
       loop != (userLoop = domOp->getParentOfType<LoopLikeOpInterface>());)
    domOp = userLoop;
  assert(loop->isProperAncestor(domOp));

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);
  Value trueVal = b.create<arith::ConstantOp>(b.getBoolAttr(true));

  Value userPred = trueVal;
  Operation *parentOp = domOp;
  b.setInsertionPoint(loop.getBody()->findAncestorOpInBlock(*domOp));
  while (loop != (parentOp = parentOp->getParentOp())) {
    assert(!isa<LoopLikeOpInterface>(parentOp));
    auto ifOp = dyn_cast<scf::IfOp>(parentOp);
    if (!ifOp) {
      llvm::report_fatal_error(
          "FIXME: unsupported parent operation for MMA user");
    }
    Value cond = ifOp.getCondition();
    if (domOp->getParentRegion() == &ifOp.getElseRegion())
      cond = b.create<arith::XOrIOp>(cond, trueVal);
    userPred = b.create<arith::AndIOp>(userPred, cond);
  }

  return userPred;
}

static MemDescType getAsMutable(MemDescType type) {
  return MemDescType::get(type.getShape(), type.getElementType(),
                          type.getEncoding(), type.getMemorySpace(),
                          /*mutableMemory=*/true);
}

static void propagateMutability(Value value) {
  for (Operation *user : value.getUsers()) {
    if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      user->getResult(0).setType(
          getAsMutable(cast<MemDescType>(user->getResult(0).getType())));
      propagateMutability(user->getResult(0));
    }
  }
}

//===----------------------------------------------------------------------===//
// MMA Pipelining
//===----------------------------------------------------------------------===//

static LogicalResult pipelineMMA(scf::ForOp &loop, PipelinedMMA &mma,
                                 WarpSchedule &schedule, DominanceInfo &domInfo,
                                 PostDominanceInfo &postDomInfo) {
  ttng::MMAv5OpInterface mmaOp = mma.mmaOp;
  auto fail = [&](StringRef msg) { return emitWarning(mmaOp.getLoc(), msg); };
  Block &body = *loop.getBody();
  auto inBody = [&](Operation *op) { return body.findAncestorOpInBlock(*op); };

  // Determine if the MMA accumulator can be multibuffered.
  bool accIsMultiBuffered =
      // MMAs in subsequent iterations can be overlapped.
      !ttng::hasAccReadModifyWrite(mmaOp, loop) &&
      // The accumulator is reset at some point, thus allowing multibuffering.
      ttng::isAccMultibufferingPossible(mmaOp, loop) &&
      // The user didn't disable it with a flag.
      !getDisallowAccMultiBuffer(loop);

  // Check that the accumulator can be multi-buffered.
  ttng::TMEMAllocOp oldAllocOp =
      mmaOp.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
  if (!oldAllocOp)
    return fail("accumulator is not a TMEM alloc");
  for (Operation *user : oldAllocOp.getResult().getUsers()) {
    if (!loop->getParentRegion()->isAncestor(user->getParentRegion()))
      return fail("cannot track accumulator uses");
  }

  PartitionBuilder b(mmaOp.getLoc(), oldAllocOp);
  int numMmaStages = 1 + accIsMultiBuffered;
  ttng::TMEMAllocOp allocOp =
      createTMemAlloc(b, oldAllocOp, /*multiBuffered=*/true, numMmaStages);

  // Use placeholder values for the indices in the loop.
  loop = addIterArgsToLoop(b, loop, {b.intCst(0), b.intCst(0)});
  auto indexPhase = loop.getRegionIterArgs().take_back(2);
  BlockArgument index = indexPhase[0];
  BlockArgument phase = indexPhase[1];

  // Replace uses of the accumulator before the loop with buffer 0, and replace
  // those after the loop with the last buffer.
  Value firstView = createSingleBufferView(b, allocOp, b.intCst(0));
  b.setInsertionPointAfter(loop);
  Value lastIndex = loop.getResult(index.getArgNumber() - 1);
  Value lastPhase = loop.getResult(phase.getArgNumber() - 1);
  Value lastView = createSingleBufferView(b, allocOp, lastIndex);

  // Find users of the accumulator in the loop and sort them by program order.
  SmallVector<Operation *> usersInLoop;
  for (OpOperand &use :
       llvm::make_early_inc_range(oldAllocOp.getResult().getUses())) {
    Operation *user = use.getOwner();
    if (user->getParentRegion() == loop->getParentRegion()) {
      if (loop->isBeforeInBlock(user))
        use.set(lastView);
      else
        use.set(firstView);
    } else if (loop.getBodyRegion().isAncestor(user->getParentRegion())) {
      usersInLoop.push_back(user);
    } else {
      return fail("cannot trace accumulator use");
    }
  }
  llvm::sort(usersInLoop, [&](Operation *lhs, Operation *rhs) {
    return inBody(lhs)->isBeforeInBlock(inBody(rhs));
  });

  // Find the read and overwrite points.
  Operation *overwriteOp = nullptr, *readOp = nullptr;
  for (Operation *user : usersInLoop) {
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(user)) {
      overwriteOp = storeOp;
    } else if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(user)) {
      if (!matchPattern(mmaOp.useAccumulator(), m_One()))
        overwriteOp = mmaOp;
    } else if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(user)) {
      readOp = loadOp;
    } else {
      llvm::report_fatal_error("FIXME: unhandled MMA accumulator user");
    }
  }

  if (!overwriteOp)
    overwriteOp = mmaOp;
  if (!readOp)
    readOp = overwriteOp;

  struct Node {
    Operation *op;
    Value barPrev;
    Value barNext;
    Value index;
    Value phase;
  };

  SmallVector<Node, 3> nodes{Node{overwriteOp}, Node{mmaOp}, Node{readOp}};
  llvm::sort(nodes, [&](Node &lhs, Node &rhs) {
    return inBody(lhs.op)->isBeforeInBlock(inBody(rhs.op));
  });

  for (int i = 0; i < nodes.size(); ++i) {
    Node &cur = nodes[i];
    Node &next = nodes[(i + 1) % nodes.size()];
    if (schedule.getPartition(inBody(cur.op)) !=
        schedule.getPartition(inBody(next.op))) {
      cur.barNext = createBarrierAlloc(loop, numMmaStages);
      next.barPrev = cur.barNext;
    }
  }

  // If the first node has a barrier, fully initialize it to let it run.
  if (nodes.front().barPrev) {
    for (auto i : llvm::seq(numMmaStages)) {
      b.setInsertionPoint(loop);
      Value bar = createSingleBufferView(b, nodes.front().barPrev, i);
      b.create<ttng::ArriveBarrierOp>(bar, /*arriveCount=*/1);
    }
  }

  Value userPred = b.boolCst(true);
  if (readOp == mmaOp) {
    PartitionBuilder b(mmaOp.getLoc(), mmaOp);
    Value lastInductionValue = [&]() {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(loop);
      return getLastInductionValue(b, loop);
    }();
    userPred = b.create<arith::CmpIOp>(
        arith::CmpIPredicate::eq, loop.getInductionVar(), lastInductionValue);
    nodes.back().barNext = createBarrierAlloc(loop, /*numBarriers=*/1);
  }

  Value curIndex = index, curPhase = phase;
  b.setInsertionPoint(loop);
  Value replTok = b.create<ub::PoisonOp>(b.getType<AsyncTokenType>());
  DenseSet<Operation *> seen;
  std::optional<OpBuilder::InsertPoint> incrementPt;
  Node *firstAfterInc = nullptr;
  for (Node &node : nodes) {
    node.index = curIndex;
    node.phase = curPhase;
    if (incrementPt && node.barPrev && !firstAfterInc)
      firstAfterInc = &node;
    if (!seen.insert(node.op).second)
      continue;
    b.setInsertionPoint(node.op);
    Value view = createSingleBufferView(b, allocOp, node.index);
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(node.op)) {
      storeOp.getDstMutable().assign(view);
      storeOp.getDepMutable().clear();
      storeOp.getToken().replaceAllUsesWith(replTok);
    } else if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(node.op)) {
      loadOp.getSrcMutable().assign(view);
      loadOp.getDepMutable().clear();
      loadOp.getToken().replaceAllUsesWith(replTok);
    } else {
      assert(node.op == mmaOp);
      mmaOp.setAccumulator(view);
      mmaOp.getAccDepMutable().clear();
      mmaOp.getToken().replaceAllUsesWith(replTok);
    }
    if (node.op == dyn_cast<ttng::TMEMLoadOp>(readOp)) {
      ImplicitLocOpBuilder b(readOp->getLoc(), loop);
      userPred = getUserPrecondition(b, loop, node.op);
      b.setInsertionPointAfter(inBody(readOp));
      auto [nextIndex, nextPhase] =
          postIncrementModulo(b, index, phase, numMmaStages);
      curIndex = b.create<arith::SelectOp>(userPred, nextIndex, index);
      curPhase = b.create<arith::SelectOp>(userPred, nextPhase, phase);
      incrementPt = b.saveInsertionPoint();
    }
  }
  if (firstAfterInc) {
    b.setInsertionPoint(loop);
    if (firstAfterInc->op == mmaOp) {
      Value firstBar = createSingleBufferView(b, firstAfterInc->barPrev, 0);
      b.create<ttng::ArriveBarrierOp>(firstBar, /*arriveCount=*/1);
    } else {
      assert(firstAfterInc->op == dyn_cast<ttng::TMEMStoreOp>(overwriteOp));
      for (auto i : llvm::seq(numMmaStages)) {
        Value firstBar = createSingleBufferView(b, firstAfterInc->barPrev, i);
        b.create<ttng::ArriveBarrierOp>(firstBar, /*arriveCount=*/1);
      }
    }
  }
  oldAllocOp.getToken().replaceAllUsesWith(allocOp.getToken());
  oldAllocOp.erase();
  cast<scf::YieldOp>(loop.getBody()->getTerminator())
      .getResultsMutable()
      .append({curIndex, curPhase});

  // Find operands that need to be pipelined through shmem.
  SmallVector<Value> incomingOperands;
  llvm::append_range(incomingOperands, mmaOp->getOperands());
  SmallVector<std::pair<Operation *, Partition *>> operandDefs;
  while (!incomingOperands.empty()) {
    Value operand = incomingOperands.pop_back_val();
    if (!isa<MemDescType>(operand.getType()))
      continue;
    Operation *defOp = operand.getDefiningOp();
    if (!defOp || loop.isDefinedOutsideOfLoop(operand))
      continue;
    defOp = inBody(defOp);
    Partition *defPartition = schedule.getPartition(defOp);

    if (!defPartition || defPartition == schedule.getRootPartition()) {
      // If the MMA operand is coming from outside the loop, move the alloc out.
      auto allocOp = dyn_cast<LocalAllocOp>(defOp);
      if (allocOp && loop.isDefinedOutsideOfLoop(allocOp.getSrc()))
        allocOp->moveBefore(loop);
      continue;
    }

    if (auto allocOp = operand.getDefiningOp<LocalAllocOp>()) {
      PartitionBuilder b(allocOp.getLoc(), allocOp);
      StageCluster stageCluster = getStageCluster(allocOp);
      auto store = b.createInto<LocalStoreOp>(*defPartition, stageCluster,
                                              allocOp.getSrc(), allocOp);
      auto fence = b.createInto<ttng::FenceAsyncSharedOp>(
          *defPartition, stageCluster, /*bCluster=*/false);
      operandDefs.emplace_back(body.findAncestorOpInBlock(*store),
                               defPartition);
      operandDefs.emplace_back(body.findAncestorOpInBlock(*fence),
                               defPartition);
      allocOp->moveBefore(loop);
      allocOp->removeAttr(kPartitionAttrName);
      allocOp.getSrcMutable().clear();
      allocOp.getResult().setType(getAsMutable(allocOp.getType()));
      propagateMutability(allocOp.getResult());
    } else if (auto tmemAllocOp = operand.getDefiningOp<ttng::TMEMAllocOp>()) {
      PartitionBuilder b(tmemAllocOp.getLoc(), tmemAllocOp);
      StageCluster stageCluster = getStageCluster(tmemAllocOp);
      auto store = b.createInto<ttng::TMEMStoreOp>(
          *defPartition, stageCluster, Type(), tmemAllocOp.getResult(), Value(),
          tmemAllocOp.getSrc(), b.boolCst(true));
      operandDefs.emplace_back(body.findAncestorOpInBlock(*store),
                               defPartition);
      tmemAllocOp->moveBefore(loop);
      tmemAllocOp->removeAttr(kPartitionAttrName);
      tmemAllocOp.getSrcMutable().clear();
      tmemAllocOp.getResult().setType(getAsMutable(tmemAllocOp.getType()));
    } else if (defOp->hasTrait<OpTrait::MemDescViewTrait>()) {
      incomingOperands.push_back(defOp->getOperand(0));
    }
  }

  for (Node &node : nodes) {
    Partition *partition = schedule.getPartition(inBody(node.op));
    PartitionBuilder b(node.op->getLoc(), loop);

    SmallVector<Operation *> defs;
    defs.push_back(node.op);

    // Find operand defs that come from the same partition and incorporate them
    // in this synchronization edge.
    decltype(operandDefs) nextOperandDefs;
    for (auto &[defOp, defPartition] : operandDefs) {
      if (defPartition == partition && inBody(node.op)->isBeforeInBlock(mmaOp))
        defs.push_back(defOp);
      else
        nextOperandDefs.emplace_back(defOp, defPartition);
    }
    operandDefs = std::move(nextOperandDefs);

    Operation *domOp = findNearestCommonDominator(defs, domInfo);
    Operation *lastOp = findNearestCommonPostDominator(defs, postDomInfo);

    StageCluster nodeStageCluster = getStageCluster(node.op);
    if (node.barPrev) {
      if (!isa<ttng::TMEMLoadOp>(node.op)) {
        // If the user precondition is defined after the MMA, we need to peel
        // the wait for the user.
        if (incrementPt && domOp->isBeforeInBlock(&*incrementPt->getPoint()) &&
            domInfo.properlyDominates(mmaOp, userPred.getDefiningOp())) {
          b.restoreInsertionPoint(*incrementPt);
          Value bar = createSingleBufferView(b, node.barPrev, curIndex);
          b.createInto<ttng::WaitBarrierOp>(*partition, nodeStageCluster, bar,
                                            curPhase, userPred);
        } else {
          b.setInsertionPoint(domOp);
          Value bar = createSingleBufferView(b, node.barPrev, node.index);
          b.createInto<ttng::WaitBarrierOp>(*partition, nodeStageCluster, bar,
                                            node.phase, userPred);
        }
      } else {
        b.setInsertionPoint(domOp);
        if (isa<scf::IfOp>(domOp->getParentOp()))
          b.setInsertionPointToStart(domOp->getBlock());
        Value bar = createSingleBufferView(b, node.barPrev, node.index);
        b.createInto<ttng::WaitBarrierOp>(*partition, nodeStageCluster, bar,
                                          node.phase);
      }
    }
    if (node.barNext) {
      if (mmaOp == node.op) {
        b.setInsertionPoint(mmaOp);
        Value bar = createSingleBufferView(b, node.barNext, node.index);
        mmaOp.addCompletionBarrier(bar, userPred);
      } else {
        b.setInsertionPointAfter(lastOp);
        if (isa<scf::IfOp>(lastOp->getParentOp()))
          b.setInsertionPoint(lastOp->getBlock()->getTerminator());
        Value bar = createSingleBufferView(b, node.barNext, node.index);
        b.createInto<ttng::ArriveBarrierOp>(*partition, nodeStageCluster, bar,
                                            1);
      }
    }
  }

  // Handle leftover operand defs.
  llvm::MapVector<Partition *, SmallVector<Operation *>> operandDefsMap;
  for (auto &[defOp, defPartition] : operandDefs)
    operandDefsMap[defPartition].push_back(defOp);
  for (auto &[partition, defs] : operandDefsMap) {
    Value emptyBar = createBarrierAlloc(loop, /*numBarriers=*/1);
    Value readyBar = createBarrierAlloc(loop, /*numBarriers=*/1);
    PartitionBuilder b(defs.front()->getLoc(), loop);
    b.create<ttng::ArriveBarrierOp>(emptyBar, /*arriveCount=*/1);

    Operation *domOp = findNearestCommonDominator(defs, domInfo);
    Operation *lastOp = findNearestCommonPostDominator(defs, postDomInfo);

    auto [index, phase] = addIndexAndPhase(b, loop, /*numStages=*/1);
    StageCluster srcStageCluster = getStageCluster(domOp);
    b.setInsertionPoint(domOp);
    b.createInto<ttng::WaitBarrierOp>(*partition, srcStageCluster, emptyBar,
                                      phase);

    b.setInsertionPointAfter(lastOp);
    b.createInto<ttng::ArriveBarrierOp>(*partition, srcStageCluster, readyBar,
                                        1);

    b.setInsertionPoint(mmaOp);
    b.createInto<ttng::WaitBarrierOp>(*schedule.getPartition(mmaOp),
                                      getStageCluster(mmaOp), readyBar, phase);
    mmaOp.addCompletionBarrier(emptyBar, b.boolCst(true));
  }

  if (nodes.back().barNext) {
    b.setInsertionPointAfter(loop);
    // Re-acquire loop results as they may have been invalidated.
    Value lastIndex = loop.getResult(index.getArgNumber() - 1);
    Value lastPhase = loop.getResult(phase.getArgNumber() - 1);
    Value lastBar = createSingleBufferView(b, nodes.back().barNext, lastIndex);
    b.create<ttng::WaitBarrierOp>(lastBar, lastPhase);
  }

  llvm::SetVector<Operation *> predOps;
  Operation *hoistPt =
      findNearestCommonDominator(llvm::to_vector(userPred.getUsers()), domInfo);
  if (!hoistPt)
    return success();
  if (!getDominatingValueSetOpsToHoist(
          domInfo, body.findAncestorOpInBlock(*hoistPt), userPred, predOps))
    return fail("failed to hoist predicate ops above MMA");
  hoistOpsBefore(hoistPt, predOps);
  return success();
}

//===----------------------------------------------------------------------===//
// lowerLoops
//===----------------------------------------------------------------------===//

LogicalResult lowerLoops(scf::ForOp &loop, MutableArrayRef<PipelinedMMA> mmas,
                         WarpSchedule &schedule, int numLoadStages) {
  Block &body = *loop.getBody();
  DominanceInfo domInfo(loop);
  PostDominanceInfo postDomInfo(loop);
  // Multi-buffer and lower the MMAs.
  for (PipelinedMMA &mma : mmas) {
    if (failed(pipelineMMA(loop, mma, schedule, domInfo, postDomInfo)))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPULOADMMASPECIALIZATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct LoadMMASpecialization
    : public triton::gpu::impl::TritonGPULoadMMASpecializationBase<
          LoadMMASpecialization> {
  using TritonGPULoadMMASpecializationBase::TritonGPULoadMMASpecializationBase;

  void runOnOperation() override;
};
} // namespace

void LoadMMASpecialization::runOnOperation() {
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName))
      loops.push_back(loop);
  });
  for (scf::ForOp loop : loops) {
    FailureOr<WarpSchedule> schedule = WarpSchedule::deserialize(loop);
    if (failed(schedule))
      continue;
    auto mmas = getPartitionScheme(loop, *schedule);
    if (mmas.empty())
      continue;
    int loopNumStages = getNumStagesOrDefault(loop, numStages);
    if (failed(lowerLoops(loop, mmas, *schedule, loopNumStages)))
      continue;
  }
}
