#!/usr/bin/env python3
import onnx
import numpy as np
from onnx import helper, numpy_helper
import sys

in_model = sys.argv[1]
out_model = sys.argv[2]

m = onnx.load(in_model)
g = m.graph

# ---- remove old broken nodes safely ----
nodes_to_keep = []
for n in g.node:
    if n.name not in ["Insert_Standardize_Sub", "Insert_Standardize_Div"]:
        nodes_to_keep.append(n)

del g.node[:]
g.node.extend(nodes_to_keep)

# ---- head input tensor (confirmed earlier) ----
HEAD_INPUT = "model/dense/BiasAdd:0"

# ---- load calibration stats ----
mean = np.load("emb_mean.npy").astype(np.float32).reshape(1, -1)
std  = np.load("emb_std.npy").astype(np.float32).reshape(1, -1)

mean_init = numpy_helper.from_array(mean, name="calib_emb_mean")
std_init  = numpy_helper.from_array(std,  name="calib_emb_std")

# remove any existing mean/std initializers
inits_to_keep = []
for init in g.initializer:
    if init.name not in ["calib_emb_mean", "calib_emb_std"]:
        inits_to_keep.append(init)

del g.initializer[:]
g.initializer.extend(inits_to_keep)
g.initializer.extend([mean_init, std_init])

# ---- create Sub and Div ----
sub_out = "emb_sub_std"
div_out = "emb_stdized"

sub_node = helper.make_node(
    "Sub",
    inputs=[HEAD_INPUT, "calib_emb_mean"],
    outputs=[sub_out],
    name="Insert_Standardize_Sub"
)

div_node = helper.make_node(
    "Div",
    inputs=[sub_out, "calib_emb_std"],
    outputs=[div_out],
    name="Insert_Standardize_Div"
)

g.node.append(sub_node)
g.node.append(div_node)

# ---- redirect head first layer to standardized embedding ----
for node in g.node:
    if node.name == "head_model/dense/MatMul":
        node.input[0] = div_out

onnx.save(m, out_model)
print("Saved fixed model to", out_model)