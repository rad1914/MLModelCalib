# @path: inject_standardizer.py
import argparse, sys, os, onnx, numpy as np
from onnx import helper, numpy_helper as nh, TensorProto as T
p = argparse.ArgumentParser()
p.add_argument('-m'); p.add_argument('--mean'); p.add_argument('--std'); p.add_argument('-o'); p.add_argument('--head')
a = p.parse_args()
if not all(map(os.path.isfile, [a.m, a.mean, a.std])): sys.exit("missing file")
m = onnx.load(a.m); g = m.graph; n = list(g.node)
h = a.head or next((i.input[0] for i in n if i.op_type.lower()=='tanh' and i.input), None)
if not h: sys.exit("no head")
mean = np.load(a.mean).astype(np.float32).ravel()
std  = np.load(a.std ).astype(np.float32).ravel()
if mean.shape != std.shape: sys.exit("shape mismatch")
g.initializer += [nh.from_array(mean,"mean"), nh.from_array(std,"std")]
i = next((k for k,x in enumerate(n) if h in x.input), len(n))
n.insert(i, helper.make_node('Sub',[h,"mean"],["sub"]))
i = next((k for k,x in enumerate(n) if "sub" in x.input), len(n))
n.insert(i, helper.make_node('Div',["sub","std"],["out"]))
for x in n: x.input[:] = ["out" if j==h else j for j in x.input]
for o in g.output: o.name = "out" if o.name==h else o.name
g.node[:] = n
g.value_info += [
    helper.make_tensor_value_info("sub", T.FLOAT, ['N', mean.size]),
    helper.make_tensor_value_info("out", T.FLOAT, ['N', mean.size])
]
onnx.save(m, a.o)
