#!/usr/bin/env python3
import onnx
import sys

m = onnx.load(sys.argv[1])

print("Looking for Insert_Standardize_Sub node...\n")

for node in m.graph.node:
    if node.name == "Insert_Standardize_Sub":
        print("Found node:")
        print("  op:", node.op_type)
        print("  inputs:", node.input)
        print("  outputs:", node.output)
        target_input = node.input[0]
        break
else:
    print("Sub node not found")
    sys.exit(0)

print("\nSearching for shape info of:", target_input)

for v in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input):
    if v.name == target_input:
        try:
            dims = [d.dim_value if d.dim_value > 0 else None
                    for d in v.type.tensor_type.shape.dim]
            print("Shape from value_info:", dims)
        except:
            print("No shape info available")