// RUN: triton-opt %s | FileCheck %s

// Test reduce operation with initial value for mixed-precision accumulation

// CHECK-LABEL: @reduce_with_init_f32_acc
tt.func @reduce_with_init_f32_acc(%v: tensor<128xf16>, %init: f32) -> f32 {
  // Reduce f16 tensor with f32 initial value - enables mixed-precision accumulation
  // The combine function receives (acc: f32, val: f16) and returns f32

  // CHECK: tt.reduce
  // CHECK-SAME: axis = 0
  // CHECK: arith.extf
  // CHECK: arith.addf
  // CHECK: tt.reduce.return
  // CHECK: (tensor<128xf16>, f32) -> f32
  %result = "tt.reduce"(%v, %init) <{axis = 0 : i32, operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%acc: f32, %val: f16):
    %val_f32 = arith.extf %val : f16 to f32
    %sum = arith.addf %acc, %val_f32 : f32
    tt.reduce.return %sum : f32
  }) : (tensor<128xf16>, f32) -> f32
  tt.return %result : f32
}

// CHECK-LABEL: @reduce_with_init_2d
tt.func @reduce_with_init_2d(%v: tensor<32x64xf16>, %init: f32) -> tensor<64xf32> {
  // Reduce along axis 0 with f32 accumulator

  // CHECK: tt.reduce
  // CHECK-SAME: axis = 0
  // CHECK: (tensor<32x64xf16>, f32) -> tensor<64xf32>
  %result = "tt.reduce"(%v, %init) <{axis = 0 : i32, operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%acc: f32, %val: f16):
    %val_f32 = arith.extf %val : f16 to f32
    %sum = arith.addf %acc, %val_f32 : f32
    tt.reduce.return %sum : f32
  }) : (tensor<32x64xf16>, f32) -> tensor<64xf32>
  tt.return %result : tensor<64xf32>
}

// CHECK-LABEL: @reduce_with_init_same_type
tt.func @reduce_with_init_same_type(%v: tensor<128xf32>, %init: f32) -> f32 {
  // Reduce with initial value of same type as input

  // CHECK: tt.reduce
  // CHECK-SAME: axis = 0
  // CHECK: arith.addf
  // CHECK: tt.reduce.return
  // CHECK: (tensor<128xf32>, f32) -> f32
  %result = "tt.reduce"(%v, %init) <{axis = 0 : i32, operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%acc: f32, %val: f32):
    %sum = arith.addf %acc, %val : f32
    tt.reduce.return %sum : f32
  }) : (tensor<128xf32>, f32) -> f32
  tt.return %result : f32
}

// CHECK-LABEL: @reduce_without_init
tt.func @reduce_without_init(%v: tensor<128xf32>) -> f32 {
  // Traditional reduce without initial value (backward compatibility)

  // CHECK: tt.reduce
  // CHECK-SAME: axis = 0
  // CHECK: arith.addf
  // CHECK: tt.reduce.return
  // CHECK: (tensor<128xf32>) -> f32
  %result = "tt.reduce"(%v) <{axis = 0 : i32, operandSegmentSizes = array<i32: 1, 0>}> ({
  ^bb0(%acc: f32, %val: f32):
    %sum = arith.addf %acc, %val : f32
    tt.reduce.return %sum : f32
  }) : (tensor<128xf32>) -> f32
  tt.return %result : f32
}

// CHECK-LABEL: @reduce_with_init_i32
tt.func @reduce_with_init_i32(%v: tensor<64xi16>, %init: i32) -> i32 {
  // Integer reduction with wider accumulator

  // CHECK: tt.reduce
  // CHECK-SAME: axis = 0
  // CHECK: (tensor<64xi16>, i32) -> i32
  %result = "tt.reduce"(%v, %init) <{axis = 0 : i32, operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%acc: i32, %val: i16):
    %val_i32 = arith.extsi %val : i16 to i32
    %sum = arith.addi %acc, %val_i32 : i32
    tt.reduce.return %sum : i32
  }) : (tensor<64xi16>, i32) -> i32
  tt.return %result : i32
}


