# @path: merging.py
import sys, onnx, numpy as np
from onnx import helper, numpy_helper as nh
from onnx.compose import merge_models, add_prefix
def dims(v):
    return [d.dim_value if d.dim_value else None for d in v.type.tensor_type.shape.dim]
def pick_tensor(m, want=200):
    m = onnx.shape_inference.infer_shapes(m)
    for vi in list(m.graph.value_info) + list(m.graph.output):
        shp = dims(vi)
        if shp and shp[-1] == want:
            return vi.name
    raise SystemExit(f"no {want}-d tensor found in encoder")
a = sys.argv
if len(a) < 6: sys.exit("Usage: merge_and_stitch.py ENC HEAD MEAN STD OUT [--prefix p] [--tanh]")
p = a[a.index("--prefix")+1] if "--prefix" in a else "h_"
enc = onnx.load(a[1])
head = add_prefix(onnx.load(a[2]), p)
eo = pick_tensor(enc, 200)
hi = next(i.name for i in head.graph.input if dims(i)[-1] == 200)
enc_outs = {o.name for o in enc.graph.output}
if eo not in enc_outs:
    from onnx import helper, TensorProto
    enc.graph.output.append(
        helper.make_tensor_value_info(eo, TensorProto.FLOAT, None)
    )
m = merge_models(enc, head, io_map=[(eo, hi)])
g = m.graph
g.initializer.extend([
    nh.from_array(np.load(a[3]).astype(np.float32), "mean"),
    nh.from_array(np.load(a[4]).astype(np.float32), "std"),
])
sub, div = eo+"_sub", eo+"_div"
i = next((i for i,n in enumerate(g.node) if hi in n.input), len(g.node))
n1 = helper.make_node("Sub", [eo, "mean"], [sub])
n2 = helper.make_node("Div", [sub, "std"], [div])
g.node.insert(i, n1)
g.node.insert(i+1, n2)
for n in g.node:
    if sub in n.output or div in n.output: continue
    n.input[:] = [div if x == hi else x for x in n.input]
if "--tanh" in a:
    o = g.output[0].name
    t = o+"_t"
    g.node.append(helper.make_node("Tanh", [o], [t]))
    g.output[0].name = t
    for v in g.value_info:
        if v.name == o: v.name = t
g.ClearField("value_info")
onnx.save(m, a[5])
