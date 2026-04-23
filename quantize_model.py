#!/usr/bin/env python3
# @path: quantize_model.py
import argparse
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_in")
    p.add_argument("model_out")
    p.add_argument("calib_dir")
    args = p.parse_args()
    quantize_dynamic(
        args.model_in,
        args.model_out,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    print("Saved:", args.model_out)
if __name__ == "__main__":
    main()
