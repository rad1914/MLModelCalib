import onnx

src = "merged_std_final.onnx"
dst = "merged_std_final_fixed.onnx"

m = onnx.load(src)

# wipe all opset imports (clean slate)
m.ClearField("opset_import")

# add proper ai.onnx domain
opset = m.opset_import.add()
opset.domain = "ai.onnx"
opset.version = 13

onnx.save(m, dst)

print("Saved:", dst)