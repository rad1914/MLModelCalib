#!/usr/bin/env python3
import onnx
from onnx import helper
import sys

in_model = sys.argv[1]
out_model = sys.argv[2]

m = onnx.load(in_model)
g = m.graph

# ---- find Tanh node ----
for node in g.node:
    if node.op_type == "Tanh":
        tanh_node = node
        break
else:
    raise RuntimeError("No Tanh node found")

print("Found Tanh node:", tanh_node.name)

# ---- restore correct input to Tanh ----
tanh_node.input[0] = "head_model/Identity:0"

print("Rewired Tanh input to head_model/Identity:0")

# ---- ensure graph output is correct ----
# Clear existing outputs
del g.output[:]

# Add correct final output (2-dim)
g.output.extend([
    helper.make_tensor_value_info(
        "final_output",
        onnx.TensorProto.FLOAT,
        None
    )
])

onnx.save(m, out_model)
print("Saved fixed model to", out_model)