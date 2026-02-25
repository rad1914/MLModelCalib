#!/usr/bin/env python3
# fix_opset.py
import onnx
import sys

if len(sys.argv) != 3:
    print("Usage: fix_opset.py input.onnx output.onnx")
    sys.exit(1)

inp = sys.argv[1]
out = sys.argv[2]

model = onnx.load(inp)

unique = {}
for oi in model.opset_import:
    domain = oi.domain or ""
    unique[domain] = oi.version

del model.opset_import[:]

for domain, version in unique.items():
    new_oi = model.opset_import.add()
    new_oi.domain = domain
    new_oi.version = version

onnx.save(model, out)

print("Saved fixed model:", out)
print("Opsets:")
for oi in model.opset_import:
    print(" domain:", repr(oi.domain), "version:", oi.version)