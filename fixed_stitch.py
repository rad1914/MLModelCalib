#!/usr/bin/env python3
# fixed_stitch.py
# Usage:
#   python fixed_stitch.py encoder.onnx head.onnx out.onnx [--prefix head_] [--add-tanh]
import sys
import onnx
from onnx import helper, TensorProto
from onnx.compose import merge_models, add_prefix

if len(sys.argv) < 4:
    print("Usage: python fixed_stitch.py ENCODER_ONNX HEAD_ONNX OUT_ONNX [--prefix head_] [--add-tanh]")
    sys.exit(1)

enc_path = sys.argv[1]
head_path = sys.argv[2]
out_path = sys.argv[3]
prefix = "head_"
add_tanh_flag = False
if "--prefix" in sys.argv:
    i = sys.argv.index("--prefix")
    if i+1 < len(sys.argv):
        prefix = sys.argv[i+1]
if "--add-tanh" in sys.argv:
    add_tanh_flag = True

print("Loading encoder:", enc_path)
enc = onnx.load(enc_path)
print("Loading head:", head_path)
head = onnx.load(head_path)

print("Adding prefix to head graph names to avoid collisions:", prefix)
head_pref = add_prefix(head, prefix)

enc_out_name = enc.graph.output[0].name
head_in_name = head_pref.graph.input[0].name

print(f"Encoder output name: {enc_out_name}")
print(f"Prefixed head input name: {head_in_name}")
print("Merging graphs (this wires the encoder output -> head input)...")
merged = merge_models(enc, head_pref, io_map=[(enc_out_name, head_in_name)])

# If head's output is pre-activation and you want final tanh clamp, append it:
if add_tanh_flag:
    # assume single model output; take the first output name
    if len(merged.graph.output) != 1:
        print("Warning: merged graph has", len(merged.graph.output), "outputs. Attempting to append tanh to the first one.")
    old_output = merged.graph.output[0].name
    tanh_name = old_output + "_tanh"
    print("Appending Tanh node. Old output:", old_output, "-> new:", tanh_name)
    tanh_node = helper.make_node("Tanh", inputs=[old_output], outputs=[tanh_name], name="Append_Tanh")
    merged.graph.node.append(tanh_node)
    # replace graph output to use tanh_name
    merged.graph.output[0].name = tanh_name
    # Keep same type information; but ensure output value_info uses tanh_name if present
    try:
        for vi in merged.graph.value_info:
            if vi.name == old_output:
                vi.name = tanh_name
    except Exception:
        pass

print("Saving merged model to:", out_path)
onnx.save(merged, out_path)
print("Done.")