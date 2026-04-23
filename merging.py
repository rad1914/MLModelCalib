# @path: merging.py
import sys, onnx, numpy as np
from onnx import helper, numpy_helper as nh
from onnx.compose import merge_models, add_prefix
a = sys.argv
len(a) < 6 and sys.exit("Usage: merge_and_stitch.py ENC HEAD MEAN STD OUT [--prefix p] [--tanh]")
p = a[a.index("--prefix")+1] if "--prefix" in a else "h_"
enc = onnx.load(a[1])
head = add_prefix(onnx.load(a[2]), p)
merged = merge_models(enc, head, io_map=[(enc.graph.output[0].name, head.graph.input[0].name)])
g = merged.graph
eo = enc.graph.output[0].name
hi = head.graph.input[0].name
g.initializer.extend([
    nh.from_array(np.load(a[3]).astype(np.float32), "mean"),
    nh.from_array(np.load(a[4]).astype(np.float32), "std"),
])
sub_out = eo + "_sub"
div_out = eo + "_div"
idx = next((i for i, n in enumerate(g.node) if hi in n.input), len(g.node))
g.node.insert(idx, helper.make_node("Sub", [eo, "mean"], [sub_out]))
g.node.insert(idx + 1, helper.make_node("Div", [sub_out, "std"], [div_out]))
for n in g.node:
    if any(o in (sub_out, div_out) for o in n.output):
        continue
    n.input[:] = [div_out if i == hi else i for i in n.input]
if "--tanh" in a:
    o = g.output[0].name
    t = o+"_t"
    g.node.append(helper.make_node("Tanh", [o], [t]))
    g.output[0].name = t
    for v in g.value_info:
        if v.name == o: v.name = t
g.ClearField("value_info")
onnx.save(merged, a[5])
