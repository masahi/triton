// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -reconcile-unrealized-casts 2>/dev/null | FileCheck %s --dump-input-context 20

#block0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#block2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#block3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#slice2d1 = #ttg.slice<{dim = 1, parent=#block2}>
#slice3d0 = #ttg.slice<{dim = 0, parent=#block3}>
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#A = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_v1_multictas
  tt.func @basic_insert_slice_async_v1_multictas(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}) {
    %off0_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice2d1>
    %off1_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice3d0>
    %off0 = tt.expand_dims %off0_ {axis = 1 : i32} : tensor<32xi32, #slice2d1> -> tensor<32x1xi32, #block2>
    %off1 = tt.expand_dims %off1_ {axis = 0 : i32} : tensor<32xi32, #slice3d0> -> tensor<1x32xi32, #block3>
    %broadcast_off0_scalar = tt.broadcast %off0 : tensor<32x1xi32, #block2> -> tensor<32x32xi32, #block2>
    %cst_scalar = arith.constant 32 : i32
    %cst = tt.splat %cst_scalar : i32 -> tensor<32x32xi32, #block2>
    %broadcast_off0_ = arith.muli %broadcast_off0_scalar, %cst : tensor<32x32xi32, #block2>
    %broadcast_off1_ = tt.broadcast %off1 : tensor<1x32xi32, #block3> -> tensor<32x32xi32, #block3>
    %broadcast_off0 = ttg.convert_layout %broadcast_off0_ : tensor<32x32xi32, #block2> -> tensor<32x32xi32, #AL>
    %broadcast_off1 = ttg.convert_layout %broadcast_off1_ : tensor<32x32xi32, #block3> -> tensor<32x32xi32, #AL>
    %off = arith.addi %broadcast_off0, %broadcast_off1 : tensor<32x32xi32, #AL>
    %a_init = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
    %a_ptr = tt.addptr %a_init, %off : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    %tensor = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #A, #smem, mutable>
    %index = arith.constant 1 : i32

    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4;"
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: nvvm.cp.async.commit.group
    %a = ttg.async_copy_global_to_local %a_ptr, %tensor : tensor<32x32x!tt.ptr<f32>, #AL> -> !ttg.memdesc<32x32xf32, #A, #smem, mutable>
    ttg.async_commit_group
    tt.return
  }
}
