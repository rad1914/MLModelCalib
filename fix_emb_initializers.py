#!/usr/bin/env python3
# fix_emb_initializers.py
import sys, os
import onnx
import numpy as np
from onnx import numpy_helper

if len(sys.argv) < 3:
    print("Usage: python fix_emb_initializers.py <in.onnx> <out.onnx>")
    sys.exit(1)

in_file = sys.argv[1]
out_file = sys.argv[2]

m = onnx.load(in_file)
graph = m.graph

names = [init.name for init in graph.initializer]
candidates = [n for n in names if n.startswith("calib_emb_mean") or n.startswith("calib_emb_std")]

if not candidates:
    print("No calib_emb_mean / calib_emb_std initializers found. Initializers present:", names)
    # still save a copy so user can inspect
    onnx.save(m, out_file)
    sys.exit(0)

for n in candidates:
    # find initializer index
    for i, init in enumerate(graph.initializer):
        if init.name == n:
            arr = numpy_helper.to_array(init)
            arr2 = arr.reshape(1, -1).astype(np.float32)  # (1, dim)
            new_init = numpy_helper.from_array(arr2, name=n)
            graph.initializer[i].CopyFrom(new_init)
            print(f"Rewrote initializer {n} to shape {arr2.shape}")
            break

# Save fixed model
onnx.save(m, out_file)
print("Saved fixed model to", out_file)