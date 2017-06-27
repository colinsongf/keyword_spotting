// Copyright © 2017 Naturali, Inc.
// All rights reserved.
// Created by Liu Jiahua (jiahua.liu@naturali.io) on 16 Jun 2017


#include <iostream>
#include <fstream>
#include <limits>
#include <math.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"


namespace tensorflow {

class PositionalEncodingOp : public OpKernel {
 public:
  explicit PositionalEncodingOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("encoding_size", &encoding_size_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* max_position;
    OP_REQUIRES_OK(ctx, ctx->input("max_position", &max_position));
    auto max_position_t = max_position->scalar<int>();

    Tensor* positional_encoding = nullptr;
    Status s = ctx->allocate_output(
        "positional_encoding",
        TensorShape({max_position_t(0), encoding_size_}),
        &positional_encoding);
    auto positional_encoding_t = positional_encoding->matrix<float>();

    for (int p = 0; p < max_position_t(0); ++p) {
      for (int i = 0; i < encoding_size_ / 2; ++i) {
        positional_encoding_t(p, 2*i) = sin(p / pow(10000.0, 2.0 * i / encoding_size_));
        positional_encoding_t(p, 2*i+1) = cos(p / pow(10000.0, 2.0 * i / encoding_size_));
      }
    }
  }

 private:
  int encoding_size_;
  TF_DISALLOW_COPY_AND_ASSIGN(PositionalEncodingOp);
};


using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("PositionalEncoding")
    .Input("max_position: int32")
    .Attr("encoding_size: int >= 1")
    .Output("positional_encoding: float32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle max_position;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &max_position));

      int32 encoding_size;
      TF_RETURN_IF_ERROR(c->GetAttr("encoding_size", &encoding_size));

      c->set_output(0, c->Matrix(InferenceContext::kUnknownDim, encoding_size));

      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("PositionalEncoding").Device(DEVICE_CPU),
                        PositionalEncodingOp);

} // namespace tensorflow

