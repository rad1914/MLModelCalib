#!/usr/bin/env python3
# @path: fix_opset.py
import onnx,sys
if len(sys.argv)!=3:exit("Usage: fix_opset.py in.onnx out.onnx")
m=onnx.load(sys.argv[1])
u={(o.domain or ""):o.version for o in m.opset_import}
m.opset_import.clear()
for d,v in u.items():
    x=m.opset_import.add();x.domain=d;x.version=v
onnx.save(m,sys.argv[2])
print("Saved:",sys.argv[2],"\nOpsets:",[(o.domain,o.version) for o in m.opset_import])
