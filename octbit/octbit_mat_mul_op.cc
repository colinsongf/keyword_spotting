/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Implements a quantized eight-bit version of the matmul operation.

#include <smmintrin.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/types.h"
#include <cmath>
#include <iostream>

namespace tensorflow {

/**
 * Octbit Matmul takes float matrix as the first input, and const weight matrix (qint8) as second input, and float as second
 * input, returns float out for now. This way, we can swap out MatMul only as it is most
 * expensive operator.
 */
template <class T>
class OctbitMatMulOp : public OpKernel {
 public:
  explicit OctbitMatMulOp(OpKernelConstruction* context)
  : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));
    OP_REQUIRES(context, transpose_b_, errors::InvalidArgument("b need to be transposed"));
    OP_REQUIRES(context, !transpose_a_, errors::InvalidArgument("a cannot to be transposed"));
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
    OP_REQUIRES_OK(context, context->GetAttr("bias", &bias));

    OP_REQUIRES(context, scale_ > 0, errors::InvalidArgument("scale has to be positive"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& x = context->input(0);
    const Tensor& w = context->input(1);

    // This is used to handled bias part on in computing
    auto biasvec = bias.vec<T>();
    auto mat = w.flat<qint8>().data();
    OP_REQUIRES(context, uint64(mat) % 32 == 0,
                errors::InvalidArgument("a" + uint64(mat)));

    // first we cast the input 1 into unit8 some how. we assume the second input is
    // BxH where B is batch size, and H is hidden size (i.e. transposed).
    OP_REQUIRES(
        context, x.dim_size(1) == w.dim_size(1),
        errors::InvalidArgument("f is not equal in filter and input"));

    OP_REQUIRES(
        context, x.dim_size(1) % 64 == 0,
        errors::InvalidArgument("we need to be 16 aligned." + x.shape().DebugString()));

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(x.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(w.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));

    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;   // value is 1
    dim_pair[0].second = transpose_b_ ? 1 : 0;   // value is 1

    int x_dim_remaining = 1 - dim_pair[0].first;
    int w_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {x.dim_size(x_dim_remaining), w.dim_size(w_dim_remaining)});

    Tensor* c = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &c));
    CHECK(c);

    auto output = c->matrix<T>();

    // first figure out the min/max value.
    auto bmat = x.matrix<T>();
    float min_value = std::numeric_limits<T>::max();
    float max_value = std::numeric_limits<T>::lowest();
    for (int i = 0; i < x.dim_size(0); i++) {
      for (int j = 0; j < x.dim_size(1); j++) {
        if (bmat(i, j) < min_value) min_value = bmat(i, j);
        if (bmat(i, j) > max_value) max_value = bmat(i, j);
      }
    }

    bool signed_flag = (min_value < 0);
    uint8* vec;
    posix_memalign((void**)&vec, 32,  x.dim_size(0) * x.dim_size(1) * sizeof(uint8));
    float scale = scale_;
    if (min_value < 0) {
      // assign b to [0, 254]
      T bscale = std::max(-min_value, max_value)/127;
      scale *= bscale;
      for (int i = 0; i < x.dim_size(0); i++) {
        int base = i * x.dim_size(1);
        for (int j = 0; j < x.dim_size(1); j++) {
          vec[base + j] = (quint8)(round(bmat(i, j)/bscale) + 127);
        }
      }
    } else {
      T bscale = max_value/254;
      scale *= bscale;
      for (int i = 0; i < x.dim_size(0); i++) {
        int base = i * x.dim_size(1);
        for (int j = 0; j < x.dim_size(1); j++) {
          vec[base + j] = (quint8)round(bmat(i, j)/bscale);
        }
      }
    }

    // now we need to initialize the output.
    for (int i = 0; i < c->dim_size(0); i++) {
      for (int j = 0; j < c->dim_size(1); j++) {
        output(i, j) = 0;
      }
    }

    // now it is time to compute
    int A = x.dim_size(0);
    int B = w.dim_size(0);
    int N = x.dim_size(1);
    for (int i = 0; i < B; ++i) {
      int wbase = i * N;
      for (int batch = 0; batch < A; ++batch) {
        int xbase = batch * N;
        __m128i sum;
        sum[0] = 0;
        sum[1] = 0;
        sum[2] = 0;
        sum[3] = 0;
        __m128i c, lo, hi;
        for (int j = 0; j < N/16; j += 4) {
          __m128i* a = (__m128i*) (vec + xbase + j*16);
          __m128i* b = (__m128i*) (mat + wbase + j*16);

          c = _mm_maddubs_epi16(a[0], b[0]);
          lo = _mm_cvtepi16_epi32(c);
          hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
          sum = _mm_add_epi32(_mm_add_epi32(lo, hi), sum);

          c = _mm_maddubs_epi16(a[1], b[1]);
          lo = _mm_cvtepi16_epi32(c);
          hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
          sum = _mm_add_epi32(_mm_add_epi32(lo, hi), sum);

          c = _mm_maddubs_epi16(a[2], b[2]);
          lo = _mm_cvtepi16_epi32(c);
          hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
          sum = _mm_add_epi32(_mm_add_epi32(lo, hi), sum);

          c = _mm_maddubs_epi16(a[3], b[3]);
          lo = _mm_cvtepi16_epi32(c);
          hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
          sum = _mm_add_epi32(_mm_add_epi32(lo, hi), sum);
        }

        const int32_t* val = (const int32_t*)&sum;
        for (unsigned int m = 0; m < sizeof(__m128i) / sizeof(int32_t); m++) {
          output(batch, i) += val[m];
        }
        if (signed_flag) {
          output(batch, i) -= biasvec(i);
        }
        output(batch, i) *= scale;
      }
    }
    free(vec);
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  float scale_;
  Tensor bias;
};

REGISTER_KERNEL_BUILDER(Name("OctbitMatMul").Device(DEVICE_CPU), OctbitMatMulOp<float>);
}  // namespace tensorflow


