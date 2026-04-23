# @path: inject_standardizer.py
import argparse, os, sys, numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto
p = argparse.ArgumentParser()
p.add_argument('-m', required=True)
p.add_argument('--mean', required=True)
p.add_argument('--std', required=True)
p.add_argument('-o', required=True)
p.add_argument('--head')
a = p.parse_args()
if not (os.path.isfile(a.m) and os.path.isfile(a.mean) and os.path.isfile(a.std)):
    sys.exit("missing file")
m = onnx.load(a.m)
g = m.graph
nodes = list(g.node)
head = a.head or next((n.input[0] for n in g.node if n.op_type.lower()=='tanh' and n.input), None)
if not head: sys.exit("no head")
mean = np.load(a.mean).astype(np.float32).ravel()
std  = np.load(a.std ).astype(np.float32).ravel()
if mean.shape != std.shape: sys.exit("shape mismatch")
g.initializer += [
    numpy_helper.from_array(mean, "mean"),
    numpy_helper.from_array(std,  "std")
]
sub, div = "sub", "out"
nodes.insert(
    next((i for i,n in enumerate(nodes) if head in n.input), len(nodes)),
    helper.make_node('Sub', [head, "mean"], [sub])
)
nodes.insert(
    next((i for i,n in enumerate(nodes) if sub in n.input), len(nodes)),
    helper.make_node('Div', [sub, "std"], [div])
)
for n in nodes:
    n.input[:] = [div if i==head else i for i in n.input]
for o in g.output:
    if o.name == head: o.name = div
g.node[:] = nodes
g.value_info += [
    helper.make_tensor_value_info(sub, TensorProto.FLOAT, ['N', mean.size]),
    helper.make_tensor_value_info(div, TensorProto.FLOAT, ['N', mean.size])
]
onnx.save(m, a.o)
