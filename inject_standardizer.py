#!/usr/bin/env python3
# inject_standardizer.py
"""
Insert (emb - mean) / std into an ONNX graph before the head.
Usage:
  python inject_standardizer.py \
    --model merged_float.onnx \
    --mean emb_mean.npy --std emb_std.npy \
    --out merged_with_std.onnx

If --head-input is omitted the script will try to detect a Tanh node
and use its first input as the head input (equivalent to your debug script).
"""
import argparse
import os
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', '-m', required=True, help='Input ONNX model (merged FP32)')
    p.add_argument('--mean', required=True, help='emb_mean.npy')
    p.add_argument('--std', required=True, help='emb_std.npy')
    p.add_argument('--out', '-o', required=True, help='Output ONNX file')
    p.add_argument('--head-input', help='(optional) name of the tensor that is input to the head (e.g. head_model/Identity:0)')
    return p.parse_args()

def load_npy_as_initializer(np_path: str, name: str):
    arr = np.load(np_path).astype(np.float32)
    # Ensure 1D
    arr = arr.reshape(-1)
    return numpy_helper.from_array(arr, name=name), arr.shape[0]

def find_tanh_input_name(graph):
    for node in graph.node:
        if node.op_type.lower() == 'tanh':
            if node.input:
                return node.input[0]
    return None

def find_first_consumer_index(nodes, tensor_name):
    for idx, node in enumerate(nodes):
        for inp in node.input:
            if inp == tensor_name:
                return idx
    return None

def main():
    args = parse_args()
    if not os.path.isfile(args.model):
        print("Model not found:", args.model, file=sys.stderr); sys.exit(2)
    if not os.path.isfile(args.mean) or not os.path.isfile(args.std):
        print("Mean/std .npy files not found", file=sys.stderr); sys.exit(3)

    model = onnx.load(args.model)
    graph = model.graph
    nodes = list(graph.node)

    # determine head input tensor name
    head_input = args.head_input
    if not head_input:
        head_input = find_tanh_input_name(graph)
        if not head_input:
            print("ERROR: could not auto-detect Tanh node. Provide --head-input", file=sys.stderr)
            sys.exit(4)
        print("Auto-detected head input tensor:", head_input)

    # Load mean/std as initializers
    mean_name = "calib_emb_mean"
    std_name  = "calib_emb_std"
    mean_init, dim_mean = load_npy_as_initializer(args.mean, mean_name)
    std_init, dim_std = load_npy_as_initializer(args.std, std_name)
    if dim_mean != dim_std:
        print("ERROR: mean and std dimension mismatch", file=sys.stderr); sys.exit(5)
    emb_dim = dim_mean
    print(f"Embedding dim detected: {emb_dim}")

    # Avoid duplicate initializers with same name
    existing = {init.name for init in graph.initializer}
    if mean_name in existing or std_name in existing:
        # rename to unique names
        i = 0
        while f"{mean_name}_{i}" in existing: i += 1
        mean_name = f"{mean_name}_{i}"
        std_name  = f"{std_name}_{i}"
        mean_init = numpy_helper.from_array(np.load(args.mean).astype(np.float32).reshape(-1), name=mean_name)
        std_init  = numpy_helper.from_array(np.load(args.std).astype(np.float32).reshape(-1), name=std_name)

    # append initializers to graph
    graph.initializer.extend([mean_init, std_init])

    # We'll create:
    #  Sub node: inputs [head_input, mean_name] -> output 'emb_sub_std'
    #  Div node: inputs ['emb_sub_std', std_name] -> output 'emb_stdized'
    emb_sub_name = "emb_sub_std"
    emb_stdized_name = "emb_stdized"

    sub_node = helper.make_node(
        'Sub',
        inputs=[head_input, mean_name],
        outputs=[emb_sub_name],
        name='Insert_Standardize_Sub'
    )
    div_node = helper.make_node(
        'Div',
        inputs=[emb_sub_name, std_name],
        outputs=[emb_stdized_name],
        name='Insert_Standardize_Div'
    )

    # Find the first consumer index of head_input so we can insert nodes before it
    consumer_idx = find_first_consumer_index(nodes, head_input)
    if consumer_idx is None:
        # If nobody consumes head_input (weird), append nodes but also make them available
        print("Warning: no consumer found for tensor", head_input, "– appending nodes at end", file=sys.stderr)
        nodes.append(sub_node)
        nodes.append(div_node)
    else:
        # Insert before the first consumer
        nodes.insert(consumer_idx, sub_node)
        nodes.insert(consumer_idx + 1, div_node)
    # Now rewire all nodes that previously consumed head_input to instead consume emb_stdized_name.
    # (But do not rewrite the Sub node's input.)
    for node in nodes:
        # skip our newly created sub/div nodes
        if node.name in ('Insert_Standardize_Sub', 'Insert_Standardize_Div'):
            continue
        new_inputs = []
        replaced = False
        for inp in node.input:
            if inp == head_input:
                new_inputs.append(emb_stdized_name)
                replaced = True
            else:
                new_inputs.append(inp)
        if replaced:
            # mutate
            del node.input[:]
            node.input.extend(new_inputs)

    # Also check graph.output and value_info for cases where head_input is used as output
    for out in graph.output:
        if out.name == head_input:
            out.name = emb_stdized_name

    # assign modified nodes back to graph
    del graph.node[:]
    graph.node.extend(nodes)

    # Optionally add value_info for the new outputs (not strictly required)
    vi_sub = helper.make_tensor_value_info(emb_sub_name, TensorProto.FLOAT, ['N', emb_dim])
    vi_std = helper.make_tensor_value_info(emb_stdized_name, TensorProto.FLOAT, ['N', emb_dim])
    # Avoid duplicate names in value_info
    existing_vi_names = {v.name for v in graph.value_info}
    if emb_sub_name not in existing_vi_names:
        graph.value_info.extend([vi_sub])
    if emb_stdized_name not in existing_vi_names:
        graph.value_info.extend([vi_std])

    # Save
    onnx.save(model, args.out)
    print("Saved model with inserted standardizer to:", args.out)
    print("Remember: re-run your debug/verify scripts on the new model and re-quantize after this change.")

if __name__ == '__main__':
    main()