import onnx
import sys

m = onnx.load(sys.argv[1])

print("Searching for Tanh nodes:\n")

for node in m.graph.node:
    if node.op_type == "Tanh":
        print("Node:", node.name)
        print("  inputs:", node.input)
        print("  outputs:", node.output)
        print()