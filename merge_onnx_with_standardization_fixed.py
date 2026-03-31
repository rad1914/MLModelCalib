# @path: merge_onnx_with_standardization_fixed.py

import sys
import onnx
from onnx import TensorProto
import numpy as np
from onnx import helper, numpy_helper, ModelProto

def main():
    if len(sys.argv) != 6:
        print("Usage: python merge_onnx_with_standardization_fixed.py encoder.onnx head.onnx emb_mean.npy emb_std.npy out_merged.onnx")
        sys.exit(1)

    enc_path, head_path, mean_path, std_path, out_path = sys.argv[1:]

    enc = onnx.load(enc_path)
    head = onnx.load(head_path)

    if len(enc.graph.output) != 1:
        raise RuntimeError(f"Encoder must have exactly 1 output, got {len(enc.graph.output)}")
    if len(head.graph.input) != 1:
        raise RuntimeError(f"Head must have exactly 1 input, got {len(head.graph.input)}")
    if len(head.graph.output) != 1:
        raise RuntimeError(f"Head must have exactly 1 output, got {len(head.graph.output)}")

    enc_out = enc.graph.output[0].name
    head_in  = head.graph.input[0].name

    mean = np.load(mean_path).astype(np.float32)
    std  = np.load(std_path).astype(np.float32)

    if mean.shape != std.shape:
        raise RuntimeError(f"Mean/std shape mismatch: {mean.shape} vs {std.shape}")
    if mean.ndim != 1:
        raise RuntimeError(f"Expected 1D embedding stats, got mean.ndim={mean.ndim}")
    if mean.size != 200:
        raise RuntimeError(f"Expected 200-d embedding stats, got {mean.size}")

    std = np.maximum(std, 1e-6).astype(np.float32)

    merged = ModelProto()
    merged.CopyFrom(enc)

    if enc_out == head_in:
        raise RuntimeError("Encoder output name collides with head input (will silently break graph)")

    mean_const = numpy_helper.from_array(mean.reshape(1, -1), name="std_mean_const")
    std_const  = numpy_helper.from_array(std.reshape(1, -1),  name="std_std_const")

    merged.graph.initializer.extend([mean_const, std_const])

    sub_out = enc_out + "_sub"
    div_out = enc_out + "_stded"

    sub_node = helper.make_node("Sub", [enc_out, "std_mean_const"], [sub_out], name="std_sub")
    div_node = helper.make_node("Div", [sub_out, "std_std_const"],  [div_out], name="std_div")
    merged.graph.node.extend([sub_node, div_node])

    prefix = "head_"
    all_names = set()

    for n in head.graph.node:
        all_names.update(n.input)
        all_names.update(n.output)

    for vi in head.graph.value_info:
        all_names.add(vi.name)

    for o in head.graph.output:
        all_names.add(o.name)

    all_names.discard(head_in)

    name_map = {name: prefix + name for name in all_names}

    def map_name(n):
        return name_map.get(n, n)

    head_internal = set()
    for node in head.graph.node:
        head_internal.update(node.input)
        head_internal.update(node.output)
    if head_in not in head_internal:
        raise RuntimeError(f"Head input '{head_in}' not found in head graph")
    
    for init in head.graph.initializer:
        arr = numpy_helper.to_array(init)
        new_name = map_name(init.name)
        merged.graph.initializer.append(numpy_helper.from_array(arr, name=new_name))

    for node in head.graph.node:
        mapped_inputs = [div_out if inp == head_in else map_name(inp) for inp in node.input]
        mapped_outputs = [map_name(out) for out in node.output]
        new_node = helper.make_node(
            node.op_type,
            mapped_inputs,
            mapped_outputs,
            name=prefix + (node.name or node.op_type)
        )
        for attr in node.attribute:
            new_node.attribute.extend([attr])
        merged.graph.node.append(new_node)

    sub_nodes = [n for n in merged.graph.node if n.op_type == "Sub" and len(n.output) == 1 and n.output[0] == sub_out]
    div_nodes = [n for n in merged.graph.node if n.op_type == "Div" and len(n.output) == 1 and n.output[0] == div_out]
    if not sub_nodes or not div_nodes:
        raise RuntimeError("Failed to insert standardization nodes")
    if enc_out not in sub_nodes[0].input:
        raise RuntimeError("Encoder output is not feeding the Sub node")
    if sub_out not in div_nodes[0].input:
        raise RuntimeError("Sub output is not feeding the Div node")
    head_consumers = [n for n in merged.graph.node if n.name.startswith(prefix) and div_out in n.input]
    if not head_consumers:
        raise RuntimeError("Standardized embedding does not reach the head")

    for vi in head.graph.value_info:
        new_vi = onnx.ValueInfoProto()
        new_vi.CopyFrom(vi)
        new_vi.name = map_name(vi.name)
        merged.graph.value_info.append(new_vi)

    merged.graph.output.clear()
    for o in head.graph.output:
        new_o = onnx.ValueInfoProto()
        new_o.CopyFrom(o)
        new_o.name = map_name(o.name)
        merged.graph.output.append(new_o)

    if len(merged.graph.output) != len(head.graph.output):
        raise RuntimeError("Failed to propagate head outputs into merged graph")

    def _dims(vi):
        return [
            d.dim_value if d.HasField("dim_value") else None
            for d in vi.type.tensor_type.shape.dim
        ]

    out_dims = _dims(merged.graph.output[0])
    known_out_dims = [d for d in out_dims if d is not None]
    if known_out_dims and 2 not in known_out_dims:
        raise RuntimeError(f"Head output does not look like a 2-value VA tensor: {out_dims}")

    try:
        inferred = onnx.shape_inference.infer_shapes(merged)
        onnx.checker.check_model(inferred)
        onnx.save(inferred, out_path)
        print("OK: merged + standardized ONNX saved to", out_path)
    except Exception as e:
        print("VALIDATION FAILED:", e)
        bad = out_path + ".raw.onnx"
        onnx.save(merged, bad)
        print("Saved raw for debugging:", bad)
        raise

if __name__ == "__main__":
    main()
