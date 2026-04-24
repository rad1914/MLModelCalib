#!/usr/bin/env python3
# @path: quantize_model.py
import sys, onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    sys.argv[1], sys.argv[2],
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul","Gemm"],
    extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
)
print("Saved:", sys.argv[2])
