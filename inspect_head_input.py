import onnx
import sys

m = onnx.load(sys.argv[1])

print("Searching for nodes around head_model...\n")

for node in m.graph.node:
    if "head_model" in node.name or "head_model" in " ".join(node.input):
        print("Node:", node.name)
        print("  op:", node.op_type)
        print("  inputs:", node.input)
        print("  outputs:", node.output)
        print()