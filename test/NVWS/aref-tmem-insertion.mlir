// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -nvws-insert-tmem-aref -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 64], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#shared5 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = false>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c64_i32 : i32
      %3 = tt.descriptor_load %arg3[%arg1, %2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg4[%arg2, %2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %7 = ttg.memdesc_trans %6 {order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %8 = ttng.tc_gen5_mma %5, %7, %result[%arg6], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %8 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
  tt.func @unsupported_load() {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg0 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg1 = %0) -> (!ttg.async.token)  : i32 {
      %2:2 = "get_ptrs"(%arg0) : (i32) -> (tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<64x128x!tt.ptr<f16>, #blocked1>)
      %3 = tt.load %2#0 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %4 = tt.load %2#1 : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %5 = ttg.local_alloc %3 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg1], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %7 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
  tt.func @cant_pipeline_mma(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token) -> (!ttg.async.token)  : i32 {
      %1:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %2 = tt.descriptor_load %arg0[%1#0, %1#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %3 = tt.descriptor_load %arg1[%1#1, %1#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %4 = ttg.local_alloc %2 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %5 = ttg.local_alloc %3 {ttg.partition = 2 : i32} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %6 = ttng.tmem_store %cst, %result[%arg3], %true {ttg.partition = 1 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %7 = ttng.tc_gen5_mma %4, %5, %result[%6], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %7 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }
  tt.func @invalid_acc_reset(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = 2 : i32} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %7 = ttng.tmem_store %cst, %result[%arg3], %true {ttg.partition = 1 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = ttng.tc_gen5_mma %5, %6, %result[%7], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %8 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 3 : i32}
    tt.return
  }
  tt.func @matmul_tma_acc_with_unconditional_user(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst_0, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = 2 : i32} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg3], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_1, %token_2 = ttng.tmem_load %result[%7] {ttg.partition = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "acc_user"(%result_1) {ttg.partition = 0 : i32} : (tensor<128x128xf32, #blocked>) -> ()
      %8 = ttng.tmem_store %cst, %result[%token_2], %true {ttg.partition = 1 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %8 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 4 : i32}
    tt.return
  }
  tt.func @matmul_tma_acc_with_conditional_user(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst_0, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = 2 : i32} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg3], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 : i32
      %9 = scf.if %8 -> (!ttg.async.token) {
        %result_1, %token_2 = ttng.tmem_load %result[%7] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_1) : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield %token_2 : !ttg.async.token
      } else {
        scf.yield %7 : !ttg.async.token
      } {ttg.partition = 0 : i32}
      %10 = ttng.tmem_store %cst, %result[%9], %true {ttg.partition = 1 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %10 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32}
    tt.return
  }
  tt.func @matmul_tma_acc_with_conditional_def(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = 2 : i32} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg3], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 : i32
      %result_0, %token_1 = ttng.tmem_load %result[%7] {ttg.partition = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "acc_user"(%result_0) {ttg.partition = 0 : i32} : (tensor<128x128xf32, #blocked>) -> ()
      %9 = ttng.tmem_store %cst, %result[%token_1], %8 {ttg.partition = 1 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %9 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 6 : i32}
    tt.return
  }
  tt.func @matmul_tma_acc_with_conditional_def_and_use(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = 2 : i32} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg3], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 : i32
      %9 = scf.if %8 -> (!ttg.async.token) {
        %result_0, %token_1 = ttng.tmem_load %result[%7] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield %token_1 : !ttg.async.token
      } else {
        scf.yield %7 : !ttg.async.token
      } {ttg.partition = 0 : i32}
      %10 = ttng.tmem_store %cst, %result[%9], %8 {ttg.partition = 1 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %10 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 7 : i32}
    tt.return
  }
  tt.func @matmul_tma_acc_with_conditional_def_and_use_no_multibuf_flag(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1:2 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %0) -> (i1, !ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = 2 : i32} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg4], %arg3, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 : i32
      %9 = arith.cmpi ne, %arg2, %c0_i32 : i32
      %10 = scf.if %8 -> (!ttg.async.token) {
        "some_op"() : () -> ()
        %result_0, %token_1 = ttng.tmem_load %result[%7] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield %token_1 : !ttg.async.token
      } else {
        scf.yield %7 : !ttg.async.token
      } {ttg.partition = 0 : i32}
      scf.yield %9, %10 : i1, !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 8 : i32}
    tt.return
  }
  tt.func @matmul_scaled_rhs_scales_tma(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared2>>, %arg4: !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared2>>, %arg5: !tt.tensordesc<tensor<128x8xi8, #shared3>>) {
    %cst = arith.constant dense<127> : tensor<128x8xi8, #linear>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %result = ttng.tmem_alloc %cst : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    %result_1, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst_0, %result_1[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg6 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg7 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg6, %c64_i32 : i32
      %3 = tt.descriptor_load %arg3[%arg1, %2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared2>> -> tensor<128x64xf8E4M3FN, #blocked1>
      %4 = tt.descriptor_load %arg4[%arg2, %2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared2>> -> tensor<128x64xf8E4M3FN, #blocked1>
      %5 = tt.descriptor_load %arg5[%arg1, %c0_i32] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x8xi8, #shared3>> -> tensor<128x8xi8, #linear>
      %6 = ttg.local_alloc %3 {ttg.partition = 2 : i32} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem>
      %7 = ttg.local_alloc %4 {ttg.partition = 2 : i32} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem>
      %8 = ttg.memdesc_trans %7 {order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem> -> !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>
      %result_2 = ttng.tmem_alloc %5 {ttg.partition = 2 : i32} : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      %9 = ttng.tc_gen5_mma_scaled %6, %8, %result_1[%arg7], %result, %result_2, %true, %true lhs = e4m3 rhs = e4m3 {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem>, !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      scf.yield %9 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 9 : i32}
    tt.return
  }
  tt.func @warp_specialize_only_rhs_is_loaded(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = tt.descriptor_load %arg3[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
    %1 = ttg.local_alloc %0 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %2 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %3 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %2) -> (!ttg.async.token)  : i32 {
      %4 = arith.muli %arg5, %c64_i32 : i32
      %5 = tt.descriptor_load %arg4[%arg2, %4] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %6 = ttg.local_alloc %5 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %7 = ttg.memdesc_trans %6 {order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %8 = ttng.tc_gen5_mma %1, %7, %result[%arg6], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %8 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 10 : i32}
    %result_0, %token_1 = ttng.tmem_load %result[%3] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
  tt.func @user_partition_has_cycle(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    %0 = tt.descriptor_load %arg3[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
    %1 = ttg.local_alloc %0 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %2:2 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %token) -> (tensor<128x128xf32, #blocked>, !ttg.async.token)  : i32 {
      %3 = arith.muli %arg5, %c64_i32 : i32
      %4 = tt.descriptor_load %arg4[%arg2, %3] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %5 = ttg.local_alloc %4 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.memdesc_trans %5 {order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %7 = ttng.tc_gen5_mma %1, %6, %result[%arg7], %false, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.addf %arg6, %arg6 {ttg.partition = 0 : i32} : tensor<128x128xf32, #blocked>
      %result_0, %token_1 = ttng.tmem_load %result[%7] {ttg.partition = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %9 = arith.mulf %8, %result_0 {ttg.partition = 0 : i32} : tensor<128x128xf32, #blocked>
      scf.yield %9, %token_1 : tensor<128x128xf32, #blocked>, !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 11 : i32}
    "use"(%2#0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
  tt.func @matmul_tma_acc_with_conditional_def_and_use_flag(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1:2 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %0) -> (i1, !ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = 2 : i32} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg4], %arg3, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 : i32
      %9 = arith.cmpi ne, %arg2, %c0_i32 : i32
      %10 = scf.if %8 -> (!ttg.async.token) {
        "some_op"() : () -> ()
        %result_0, %token_1 = ttng.tmem_load %result[%7] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield %token_1 : !ttg.async.token
      } else {
        scf.yield %7 : !ttg.async.token
      } {ttg.partition = 0 : i32}
      scf.yield %9, %10 : i1, !ttg.async.token
    } {tt.num_stages = 4 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 12 : i32}
    tt.return
  }
  tt.func @specialize_load_only(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      "use"(%0) {ttg.partition = 0 : i32} : (tensor<128x64xf16, #blocked1>) -> ()
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 13 : i32}
    tt.return
  }
  tt.func @fp4_padded_load(%arg0: !tt.tensordesc<tensor<1x256x64xui8, #shared5>>, %arg1: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<1x256x64xui8, #shared5>> -> tensor<256x64xi8, #blocked1>
      "use"(%0) {ttg.partition = 0 : i32} : (tensor<256x64xi8, #blocked1>) -> ()
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 14 : i32}
    tt.return
  }
  tt.func @specialize_mma_only(%arg0: !tt.tensordesc<tensor<64x128xf16, #shared>>, %arg1: !ttg.memdesc<128x64xf16, #shared, #smem>, %arg2: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%arg4 = %0) -> (!ttg.async.token)  : i32 {
      %2 = tt.descriptor_load %arg0[%arg3, %arg3] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %result_2, %token_3 = ttng.tmem_load %result[%arg4] {ttg.partition = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %3:2 = "some_producer"(%2, %result_2) {ttg.partition = 0 : i32} : (tensor<64x128xf16, #blocked1>, tensor<128x128xf32, #blocked>) -> (tensor<128x64xf16, #blocked1>, tensor<128x128xf32, #blocked>)
      %4 = ttg.local_alloc %3#0 {ttg.partition = 0 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %5 = ttg.memdesc_trans %4 {order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %6 = ttng.tmem_store %3#1, %result[%token_3], %true {ttg.partition = 0 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %7 = ttng.tc_gen5_mma %arg1, %5, %result[%6], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %7 : !ttg.async.token
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 15 : i32}
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
  tt.func @load_scale_mma_user(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem>, %arg1: !ttg.memdesc<64x128xf16, #shared, #smem>, %arg2: !tt.tensordesc<tensor<8x128xi8, #shared>>, %arg3: !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, %arg4: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg5 = %c0_i32 to %arg4 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = tt.descriptor_load %arg2[%arg5, %arg5] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<8x128xi8, #shared>> -> tensor<8x128xi8, #blocked1>
      %3 = ttg.local_alloc %2 {ttg.partition = 2 : i32} : (tensor<8x128xi8, #blocked1>) -> !ttg.memdesc<8x128xi8, #shared, #smem>
      %4 = ttg.local_load %3 {ttg.partition = 0 : i32} : !ttg.memdesc<8x128xi8, #shared, #smem> -> tensor<8x128xi8, #linear1>
      %5 = tt.trans %4 {order = array<i32: 1, 0>, ttg.partition = 0 : i32} : tensor<8x128xi8, #linear1> -> tensor<128x8xi8, #linear>
      %result_2 = ttng.tmem_alloc %5 {ttg.partition = 0 : i32} : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      %6 = ttng.tc_gen5_mma_scaled %arg0, %arg1, %result[%arg6], %result_2, %arg3, %true, %true lhs = e4m3 rhs = e4m3 {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      %result_3, %token_4 = ttng.tmem_load %result[%6] {ttg.partition = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "user"(%result_3) {ttg.partition = 0 : i32} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield %token_4 : !ttg.async.token
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 16 : i32}
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
  tt.func @store_mma_load(%arg0: i32, %arg1: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg2: !ttg.memdesc<64x128xf16, #shared, #smem>) {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = scf.for %arg3 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg4 = %token) -> (!ttg.async.token)  : i32 {
      %1 = tt.descriptor_load %arg1[%arg3, %arg3] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %2 = arith.addf %1, %1 {ttg.partition = 0 : i32} : tensor<128x64xf16, #blocked1>
      %3 = ttg.local_alloc %2 {ttg.partition = 0 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %4 = "make_acc"() : () -> tensor<128x128xf32, #blocked>
      %5 = ttng.tmem_store %4, %result[%arg4], %true {ttg.partition = 0 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %6 = ttng.tc_gen5_mma %3, %arg2, %result[%5], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_0, %token_1 = ttng.tmem_load %result[%6] {ttg.partition = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%result_0) {ttg.partition = 0 : i32} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield %token_1 : !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 17 : i32}
    tt.return
  }
  tt.func @local_alloc_into_mma(%arg0: i32, %arg1: tensor<128x64xf16, #blocked1>, %arg2: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    scf.for %arg3 = %c0_i32 to %arg0 step %c1_i32  : i32 {
      %0 = ttg.local_alloc %arg1 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %1 = tt.descriptor_load %arg2[%arg3, %arg3] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %2 = arith.addf %1, %1 {ttg.partition = 0 : i32} : tensor<64x128xf16, #blocked1>
      %3 = ttg.local_alloc %2 {ttg.partition = 0 : i32} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %4 = ttng.tc_gen5_mma %0, %3, %result[%token], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 18 : i32}
    tt.return
  }
  tt.func @shmem_sink_iterator_invalidation(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c64_i32 : i32
      %3 = tt.descriptor_load %arg4[%arg2, %2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg3[%arg1, %2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %5 = ttg.local_alloc %4 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_load %5 {ttg.partition = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked2>
      %7 = ttg.local_alloc %3 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.memdesc_trans %7 {order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %result_2 = ttng.tmem_alloc %6 {ttg.partition = 0 : i32} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory>
      %9 = ttng.tc_gen5_mma %result_2, %8, %result[%arg6], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %9 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 19 : i32}
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = false>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func public @attention_forward(%arg0: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg1: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg2: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg3: f32, %arg4: i32) {
    %cst = arith.constant dense<1.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #blocked>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_2, %token_3 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst_0, %result_2[%token_3], %true : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %1:4 = scf.for %arg5 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg6 = %cst, %arg7 = %cst_1, %arg8 = %token, %arg9 = %0) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
      %2 = tt.descriptor_load %arg1[%arg5, %c0_i32] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
      %3 = ttg.local_alloc %2 {ttg.partition = 2 : i32} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      %5 = ttng.tc_gen5_mma %arg0, %4, %result[%arg8], %false, %true {ttg.partition = 1 : i32} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_6, %token_7 = ttng.tmem_load %result[%5] {ttg.partition = 0 : i32} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
      %6 = "compute_row_max"(%result_6, %arg3) {ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %7 = "sub_row_max"(%result_6, %6, %arg3) {ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %8 = math.exp2 %7 {ttg.partition = 0 : i32} : tensor<256x64xf32, #blocked>
      %9 = arith.subf %arg7, %6 {ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %10 = arith.subf %arg7, %6 {ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %11 = math.exp2 %9 {ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %12 = math.exp2 %10 {ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %13 = "tt.reduce"(%8) <{axis = 1 : i32}> ({
      ^bb0(%arg10: f32, %arg11: f32):
        %24 = arith.addf %arg10, %arg11 : f32
        tt.reduce.return %24 : f32
      }) {ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %14 = arith.mulf %arg6, %12 {ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %15 = arith.addf %14, %13 {ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %16 = tt.expand_dims %11 {axis = 1 : i32, ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %17 = tt.broadcast %16 {ttg.partition = 3 : i32} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      %result_8, %token_9 = ttng.tmem_load %result_2[%arg9] {ttg.partition = 3 : i32} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
      %18 = arith.mulf %result_8, %17 {ttg.partition = 3 : i32} : tensor<256x64xf32, #blocked>
      %19 = tt.descriptor_load %arg2[%arg5, %c0_i32] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
      %20 = ttg.local_alloc %19 {ttg.partition = 2 : i32} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %21 = arith.truncf %8 {ttg.partition = 0 : i32} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      %result_10 = ttng.tmem_alloc %21 {ttg.partition = 0 : i32} : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory>
      %22 = ttng.tmem_store %18, %result_2[%token_9], %true {ttg.partition = 3 : i32} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %23 = ttng.tc_gen5_mma %result_10, %20, %result_2[%22], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %15, %6, %token_7, %23 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    %result_4, %token_5 = ttng.tmem_load %result_2[%1#3] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
    "use"(%1#0, %result_4, %1#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}
