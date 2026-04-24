#!/usr/bin/env python3
# @path: validation.py
import numpy as np, onnx, onnxruntime as ort, librosa, argparse
from onnx import helper, TensorProto
SR, N_FFT, HOP, N_MELS, FRAMES = 16000, 512, 256, 96, 187
def dims(v):
    return [d.dim_value if d.dim_value else None for d in v.type.tensor_type.shape.dim]
def pick_tensor(m, want=200):
    m = onnx.shape_inference.infer_shapes(m)
    for vi in list(m.graph.value_info) + list(m.graph.output):
        shp = dims(vi)
        if shp and shp[-1] == want:
            return vi.name
    raise SystemExit(f"no {want}-d tensor found in encoder")
def mel(p):
    y,_ = librosa.load(p, sr=SR, mono=True)
    m = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    ), ref=np.max).T.astype(np.float32)
    return m[:FRAMES] if len(m)>=FRAMES else np.pad(
        m, ((0,FRAMES-len(m)),(0,0)), constant_values=(m.min() if len(m) else -80)
    )
def run(p, x, out_name=None):
    s = ort.InferenceSession(p)
    outs = None if out_name is None else [out_name]
    return s.run(outs, {s.get_inputs()[0].name: x})[0]
def pick_out_2d(model_path):
    s = ort.InferenceSession(model_path)
    for o in s.get_outputs():
        shp = list(o.shape)
        if shp and shp[-1] == 2:
            return o.name
    return s.get_outputs()[-1].name
def expose_tensor(model_path, out_path, tensor_name):
    m = onnx.shape_inference.infer_shapes(onnx.load(model_path))
    if tensor_name not in {o.name for o in m.graph.output}:
        vi = next(v for v in list(m.graph.value_info) if v.name == tensor_name)
        m.graph.output.append(vi)
    onnx.save(m, out_path)
def add_dbg(i, o):
    m = onnx.load(i)
    n = m.graph.node
    sub = next((x.output[0] for x in n if x.op_type == "Sub" and x.output), None)
    div = next((x.output[0] for x in n if x.op_type == "Div" and x.output), None)
    pre = next((x.input[0] for x in reversed(n) if x.op_type == "Tanh" and x.input), None)
    hr = pre or (n[-1].output[0] if n and n[-1].output else None)
    ex = {x.name for x in m.graph.output}
    for x in (sub, div, hr):
        if x and x not in ex:
            m.graph.output.append(helper.make_tensor_value_info(x, TensorProto.FLOAT, None))
    onnx.save(m, o)
    return sub, div, hr
def main():
    a = argparse.ArgumentParser()
    a.add_argument("--audio", default="test.wav")
    a.add_argument("--enc", default="msd_musicnn.onnx")
    a.add_argument("--head", default="deam_head.onnx")
    a.add_argument("--merged", required=True)
    a.add_argument("--debug_out", default="merged_debug.onnx")
    a.add_argument("--mean", default="emb_mean.npy")
    a.add_argument("--std", default="emb_std.npy")
    a = a.parse_args()
    
    x = mel(a.audio)[None]
    M, S = np.load(a.mean), np.load(a.std)
    
    enc_name = pick_tensor(onnx.load(a.enc), 200)
    tmp_enc = a.enc + ".dbg.onnx"
    expose_tensor(a.enc, tmp_enc, enc_name)
    enc_out = run(tmp_enc, x, enc_name)
    
    d = enc_out.shape[-1]
    print("enc_shape:", enc_out.shape, "mean_shape:", M.shape, "std_shape:", S.shape)
    assert M.shape == S.shape == (d,), (M.shape, S.shape, d)
    
    z = (enc_out - M) / np.maximum(S, 1e-8)
    assert z.shape[-1] == d, z.shape
    
    r = run(a.head, z.astype(np.float32))[0]
    py = np.tanh(r)
    
    merged_out = pick_out_2d(a.merged)
    mg = run(a.merged, x, merged_out)[0]
    
    print("Diff max:", np.abs(py - mg).max())
    
    sub, div, hr = add_dbg(a.merged, a.debug_out)
    dbg = run(a.debug_out, x)
    print("Debug keys:", len(dbg), sub, div, hr)
if __name__ == "__main__": main()