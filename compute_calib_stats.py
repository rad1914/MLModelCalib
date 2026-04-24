#!/usr/bin/env python3
# @path: compute_calib_stats.py
import os, sys, argparse, numpy as np, onnxruntime as ort
import onnx
from audio_utils import load_audio_mono, make_mel_patch

def dims(v):
    return [d.dim_value if d.dim_value else None for d in v.type.tensor_type.shape.dim]

def pick_tensor(m, want=200):
    m = onnx.shape_inference.infer_shapes(m)
    for vi in list(m.graph.value_info) + list(m.graph.output):
        shp = dims(vi)
        if shp and shp[-1] == want:
            return vi.name
    raise SystemExit(f"no {want}-d tensor found in encoder")

p = argparse.ArgumentParser()
p.add_argument("-m","--model",default="msd_musicnn.onnx")
p.add_argument("-c","--calib",default="calib_wavs")
p.add_argument("--out-mean",default="emb_mean.npy")
p.add_argument("--out-std",default="emb_std.npy")
a = p.parse_args()
os.path.isfile(a.model) or sys.exit(f"Missing model: {a.model}")
os.path.isdir(a.calib) or sys.exit(f"Missing calib dir: {a.calib}")

enc = onnx.load(a.model)
tensor_name = pick_tensor(enc, 200)
tmp_model = a.model + ".calibdbg.onnx"
if tensor_name not in {o.name for o in enc.graph.output}:
    vi = onnx.shape_inference.infer_shapes(enc)
    vi = next(v for v in list(vi.graph.value_info) if v.name == tensor_name)
    enc.graph.output.append(vi)
    onnx.save(enc, tmp_model)
    s = ort.InferenceSession(tmp_model, providers=["CPUExecutionProvider"])
else:
    s = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
i = s.get_inputs()[0]

def prep(x):
    x = x.astype(np.float32)
    return x[None, ..., None] if len(i.shape) == 4 else x[None]

done_f = os.path.join(a.calib, "_processed.txt")
done = set(open(done_f).read().split()) if os.path.isfile(done_f) else set()
E = []
for f in sorted(x for x in os.listdir(a.calib) if x.lower().endswith(".wav") and x not in done):
    try:
        x = make_mel_patch(load_audio_mono(os.path.join(a.calib,f)))
        e = s.run([tensor_name], {i.name: prep(x)})[0].ravel().astype(np.float32)
        E.append(e)
        open(done_f,"a").write(f+"\n")
    except Exception as e:
        print(f"[skip] {f}: {e}", file=sys.stderr)
E or sys.exit("No valid WAVs processed")
E = np.stack(E).astype(np.float32)
std = E.std(0); std[std<1e-6]=1.0
np.save(a.out_mean, E.mean(0))
np.save(a.out_std, std)
print("Saved:", a.out_mean, E.mean(0).shape)
print("Saved:", a.out_std, std.shape)