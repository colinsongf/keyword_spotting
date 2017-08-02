#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("OctbitMatMul")
    .Input("input_a: T")
    .Input("input_b: qint8")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = true")
    .Attr("scale: float = 0.0")
    .Attr("bias: tensor")
    .Output("output: T")
    .Attr("T: type = DT_FLOAT");
