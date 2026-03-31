# @path: stitch_encoder_head.py

import onnx
from onnx import helper
from onnx.compose import merge_models
import sys

if len(sys.argv) != 5:
    print("Usage: python stitch_encoder_head.py encoder_onnx encoder_name head_onnx out_onnx")
    print("Example: python stitch_encoder_head.py merged_std_correct.onnx merged_std_correct deam_head.onnx merged_va.onnx")
    sys.exit(1)

encoder_path = sys.argv[1]

encoder_name = sys.argv[2]
head_path = sys.argv[3]
out_path = sys.argv[4]

print("Loading encoder:", encoder_path)
enc = onnx.load(encoder_path)
print("Loading head:", head_path)
head = onnx.load(head_path)

enc_out_name = enc.graph.output[0].name
head_in_name = head.graph.input[0].name

print("Connecting encoder output '{}' -> head input '{}'".format(enc_out_name, head_in_name))

merged = merge_models(enc, head, io_map=[(enc_out_name, head_in_name)])

print("Merged model has outputs:", [o.name for o in merged.graph.output])

onnx.save(merged, out_path)
print("Saved merged model:", out_path)
