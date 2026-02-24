import sys
import onnx
import numpy as np
from onnx import helper, numpy_helper, ModelProto

def main():
    if len(sys.argv) != 6:
        print("Usage: python merge_onnx_with_standardization_correct.py encoder.onnx head.onnx emb_mean.npy emb_std.npy out_merged.onnx")
        sys.exit(1)

    enc_path, head_path, mean_path, std_path, out_path = sys.argv[1:]

    enc = onnx.load(enc_path)
    head = onnx.load(head_path)

    enc_out = enc.graph.output[0].name
    head_in  = head.graph.input[0].name

    # Load stats
    mean = np.load(mean_path).astype(np.float32)
    std  = np.load(std_path).astype(np.float32)

    # ---------- Build merged base ----------
    merged = ModelProto()
    merged.CopyFrom(enc)

    mean_const = numpy_helper.from_array(mean, name="std_mean_const")
    std_const  = numpy_helper.from_array(std,  name="std_std_const")

    merged.graph.initializer.extend([mean_const, std_const])

    sub_out = enc_out + "_sub"
    div_out = enc_out + "_stded"

    sub_node = helper.make_node("Sub", [enc_out, "std_mean_const"], [sub_out], name="std_sub")
    div_node = helper.make_node("Div", [sub_out, "std_std_const"],  [div_out], name="std_div")
    merged.graph.node.extend([sub_node, div_node])

    # ---------- Collect all value names in head ----------
    # These must ALL be prefixed to avoid collisions
    prefix = "head_"
    all_names = set()

    for n in head.graph.node:
        all_names.update(n.input)
        all_names.update(n.output)

    for vi in head.graph.value_info:
        all_names.add(vi.name)

    for o in head.graph.output:
        all_names.add(o.name)

    # Do NOT prefix the head's main input
    all_names.discard(head_in)

    # Map names -> prefixed names
    name_map = {name: prefix + name for name in all_names}

    def map_name(n):
        return name_map.get(n, n)

    # ---------- Copy head initializers ----------
    for init in head.graph.initializer:
        arr = numpy_helper.to_array(init)
        new_name = map_name(init.name)
        merged.graph.initializer.append(numpy_helper.from_array(arr, name=new_name))

    # ---------- Copy head nodes with fully remapped names ----------
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

    # ---------- Copy value_info (optional but cleaner) ----------
    for vi in head.graph.value_info:
        new_vi = onnx.ValueInfoProto()
        new_vi.CopyFrom(vi)
        new_vi.name = map_name(vi.name)
        merged.graph.value_info.append(new_vi)

    # ---------- Replace merged output with prefixed head outputs ----------
    merged.graph.output.clear()
    for o in head.graph.output:
        new_o = onnx.ValueInfoProto()
        new_o.CopyFrom(o)
        new_o.name = map_name(o.name)
        merged.graph.output.append(new_o)

    # ---------- Validate ----------
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