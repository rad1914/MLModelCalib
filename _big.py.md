# @path: merging.py
import sys, onnx, numpy as np
from onnx import helper, numpy_helper as nh
from onnx.compose import merge_models, add_prefix
def dims(v):
    return [d.dim_value if d.dim_value else None for d in v.type.tensor_type.shape.dim]
def pick_tensor(m, want=200):
    m = onnx.shape_inference.infer_shapes(m)
    for vi in list(m.graph.value_info) + list(m.graph.output):
        shp = dims(vi)
        if shp and shp[-1] == want:
            return vi.name
    raise SystemExit(f"no {want}-d tensor found in encoder")
a = sys.argv
if len(a) < 6: sys.exit("Usage: merge_and_stitch.py ENC HEAD MEAN STD OUT [--prefix p] [--tanh]")
p = a[a.index("--prefix")+1] if "--prefix" in a else "h_"
enc = onnx.load(a[1])
head = add_prefix(onnx.load(a[2]), p)
eo = pick_tensor(enc, 200)
hi = next(i.name for i in head.graph.input if dims(i)[-1] == 200)
enc_outs = {o.name for o in enc.graph.output}
if eo not in enc_outs:
    from onnx import helper, TensorProto
    enc.graph.output.append(
        helper.make_tensor_value_info(eo, TensorProto.FLOAT, None)
    )
m = merge_models(enc, head, io_map=[(eo, hi)])
g = m.graph
g.initializer.extend([
    nh.from_array(np.load(a[3]).astype(np.float32), "mean"),
    nh.from_array(np.load(a[4]).astype(np.float32), "std"),
])
sub, div = eo+"_sub", eo+"_div"
i = next((i for i,n in enumerate(g.node) if eo in n.input), len(g.node))
n1 = helper.make_node("Sub", [eo, "mean"], [sub])
n2 = helper.make_node("Div", [sub, "std"], [div])
g.node.insert(i, n1)
g.node.insert(i+1, n2)
for n in g.node:
    if sub in n.output or div in n.output: continue
    n.input[:] = [div if x == eo else x for x in n.input]
if "--tanh" in a:
    m_inf = onnx.shape_inference.infer_shapes(m)
    out_vi = next(
        (ov for ov in m_inf.graph.output if dims(ov) and dims(ov)[-1] == 2),
        g.output[-1],
    )
    o = out_vi.name
    t = o + "_t"
    g.node.append(helper.make_node("Tanh", [o], [t]))
    target = next(x for x in g.output if x.name == o)
    target.name = t
g.ClearField("value_info")
onnx.save(m, a[5])
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
# @path: gen_head_calib.py
import sys, glob, os, numpy as np, librosa, onnxruntime as ort
SR,N_FFT,HOP,N_MELS,FRAMES=16000,512,256,96,187
def mel(f):
    y,_=librosa.load(f,sr=SR,mono=True)
    m=librosa.power_to_db(librosa.feature.melspectrogram(
        y=y,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS
    ),ref=np.max).T.astype('float32')
    return m[:FRAMES] if len(m)>=FRAMES else np.vstack([m,np.full((FRAMES-len(m),N_MELS),m.min() if len(m) else -80,'float32')])
len(sys.argv)!=5 and sys.exit("Usage: gen_head_calib.py ENC DIR MEAN STD")
enc,d,mp,sp=sys.argv[1:]
os.makedirs("head_calib",exist_ok=True)
mean,std=np.load(mp).astype('float32'),np.load(sp).astype('float32')
std[std==0]=1
s=ort.InferenceSession(enc,providers=["CPUExecutionProvider"])
inp=s.get_inputs()[0].name
i=-1
for i,f in enumerate(glob.glob(d+"/*.wav")):
    e=s.run(None,{inp:mel(f)[None]})[0].squeeze().astype('float32')
    np.save(f"head_calib/calib_{i:04d}.npy",(e-mean)/std)
print("Saved",i+1)
# @path: quantize_qdq_final.py
import sys, os, numpy as np
import soundfile as sf
from onnxruntime.quantization import (
     quantize_static, CalibrationDataReader,
     QuantFormat, QuantType,
     CalibrationMethod
)
class AudioCalibReader(CalibrationDataReader):
    MAX_CALIB_FILES = 120
    def __init__(self, calib_dir, model_path):
        import onnxruntime as ort
        self.idx    = 0
        _sess = ort.InferenceSession(model_path,
                                     providers=["CPUExecutionProvider"])
        self.input_name = _sess.get_inputs()[0].name
        all_files = sorted(
            os.path.join(calib_dir, f)
            for f in os.listdir(calib_dir) if f.endswith(".wav")
        )
        files = all_files[: self.MAX_CALIB_FILES]
        if not files:
            raise RuntimeError(f"No .wav files found in {calib_dir}")
        print(f"  ↳ streaming {len(files)} / {len(all_files)} mel patches ...")
        self.files = files
    def rewind(self):
        self.idx = 0
    @staticmethod
    def _mel(f, SR=16000, N_FFT=512, HOP=256, N_MELS=96, FRAMES=187):
        import librosa
        y, sr = sf.read(f, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        m = librosa.power_to_db(librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
        ), ref=np.max).T.astype(np.float32)
        if len(m) < FRAMES:
            m = np.pad(m, ((0, FRAMES - len(m)), (0, 0)), constant_values=m.min())
        return m[:FRAMES][None]
    def get_next(self):
        if self.idx >= len(self.files):
            return None
        f = self.files[self.idx]
        item = {self.input_name: self._mel(f)}
        self.idx += 1
        return item
def main():
    if len(sys.argv) != 4:
        print("Usage: quantize_qdq_final.py fp32.onnx calib_dir out.onnx")
        sys.exit(1)
    fp32, calib_dir, out = sys.argv[1:]
    reader = AudioCalibReader(calib_dir, fp32)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    quantize_static(
        model_input=fp32,
        model_output=out,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
        calibrate_method=CalibrationMethod.Entropy,
    )
    print("Saved:", out)
if __name__ == "__main__":
    main()
# @path: audio_utils.py
import numpy as np, librosa
def load_audio_mono(p, sr=16000):
    return librosa.load(p, sr=sr, mono=True)[0].astype(np.float32)
def make_mel_patch(y, sr=16000, n_fft=512, hop=256, n_mels=96, frames=187, power=2.0):
    m = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=power),
        ref=np.max
    ).T.astype(np.float32)
    if m.shape[0] >= frames:
        return m[:frames]
    pad = np.full((frames - m.shape[0], n_mels), m.min() if m.size else -80.0, np.float32)
    return np.vstack((m, pad))
def prepare_model_input(p, s):
    a = p[None].astype(np.float32)
    if not s: return a
    s = list(s)
    if len(s) == 3:
        return a if (s[1] in (None, a.shape[1]) and s[2] in (None, a.shape[2])) else a.transpose(0,2,1)
    if len(s) == 4:
        if s[1] in (None, a.shape[1]) and s[2] in (None, a.shape[2]): return a[...,None]
        if s[2] in (None, a.shape[1]) and s[3] in (None, a.shape[2]): return a[:,None]
        return a.transpose(0,2,1)[...,None]
    return a
def standardize_embedding(e, m, s, eps=1e-6):
    e = np.asarray(e, np.float32).ravel()
    m = np.asarray(m, np.float32).ravel()
    s = np.asarray(s, np.float32).ravel()
    if m.size == 1: m = np.full_like(e, m[0])
    if s.size == 1: s = np.full_like(e, s[0])
    if m.size != e.size or s.size != e.size: raise ValueError("size mismatch")
    return ((e - m) / np.where(np.abs(s) < eps, 1.0, s)).astype(np.float32)
# @path: inject_standardizer.py
import argparse, sys, os, onnx, numpy as np
from onnx import helper, numpy_helper as nh, TensorProto as T
p = argparse.ArgumentParser()
p.add_argument('-m'); p.add_argument('--mean'); p.add_argument('--std'); p.add_argument('-o'); p.add_argument('--head')
a = p.parse_args()
if not all(map(os.path.isfile, [a.m, a.mean, a.std])): sys.exit("missing file")
m = onnx.load(a.m); g = m.graph; n = list(g.node)
h = a.head or next((i.input[0] for i in n if i.op_type.lower()=='tanh' and i.input), None)
if not h: sys.exit("no head")
mean = np.load(a.mean).astype(np.float32).ravel()
std  = np.load(a.std ).astype(np.float32).ravel()
if mean.shape != std.shape: sys.exit("shape mismatch")
g.initializer += [nh.from_array(mean,"mean"), nh.from_array(std,"std")]
i = next((k for k,x in enumerate(n) if h in x.input), len(n))
n.insert(i, helper.make_node('Sub',[h,"mean"],["sub"]))
i = next((k for k,x in enumerate(n) if "sub" in x.input), len(n))
n.insert(i, helper.make_node('Div',["sub","std"],["out"]))
for x in n: x.input[:] = ["out" if j==h else j for j in x.input]
for o in g.output: o.name = "out" if o.name==h else o.name
g.node[:] = n
g.value_info += [
    helper.make_tensor_value_info("sub", T.FLOAT, ['N', mean.size]),
    helper.make_tensor_value_info("out", T.FLOAT, ['N', mean.size])
]
onnx.save(m, a.o)
#!/usr/bin/env python3
# @path: runners.py
import os, sys, json, argparse, numpy as np, soundfile as sf, librosa, onnxruntime as ort
SR,N_FFT,HOP,N_MELS,FRAMES=16000,512,256,96,187
act={"raw":lambda x:x,"tanh":np.tanh,"sigmoid":lambda x:1/(1+np.exp(-x))}
die=lambda p: os.path.exists(p) or sys.exit(f"Missing: {p}")
load=lambda p:(np.load(p) if p.endswith('.npy') else np.fromfile(p,np.float32)).astype(np.float32)
def wav(p,sr):
    die(p); y,s=sf.read(p,dtype='float32')
    return librosa.resample(y.mean(1) if y.ndim>1 else y,orig_sr=s,target_sr=sr) if s!=sr else y
def mel(y,sr,f):
    m=librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=sr,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS),ref=np.max).T.astype(np.float32)
    return m[:f] if len(m)>=f else np.vstack([m,np.full((f-len(m),N_MELS),m.min() if len(m) else -80,np.float32)])
def run(s,x):
    i=s.get_inputs()[0]
    return s.run(None,{i.name:(x.T[None] if len(i.shape)>=3 and i.shape[2]==x.shape[0] else x[None]).astype(np.float32)})[0]
p=argparse.ArgumentParser()
[p.add_argument(f"--{k}",required=True) for k in("audio","encoder","head","mean","std")]
p.add_argument("--sr",type=int,default=SR); p.add_argument("--frames",type=int,default=FRAMES)
p.add_argument("--activation",default="tanh",choices=act)
a=p.parse_args()
[die(getattr(a,k)) for k in("audio","encoder","head","mean","std")]
y=wav(a.audio,a.sr)
m=mel(y,a.sr,a.frames)
emb=run(ort.InferenceSession(a.encoder,providers=["CPUExecutionProvider"]),m).ravel()
mean,std=load(a.mean),load(a.std)
if mean.size==1: mean=np.full_like(emb,mean.item())
if std.size==1: std=np.full_like(emb,std.item())
if mean.size!=emb.size or std.size!=emb.size: sys.exit("mean/std size mismatch")
emb=(emb-mean)/(std+1e-12)
out=run(ort.InferenceSession(a.head,providers=["CPUExecutionProvider"]),emb)
print(json.dumps({"output":act[a.activation](out)[0].tolist()}))
# @path: verify_quant_final.py
import sys, os, numpy as np, librosa, onnxruntime as ort, math
from scipy.stats import pearsonr
SR,N_FFT,HOP,N_MELS,FRAMES = 16000,512,256,96,187
def mel(f):
    y,_ = librosa.load(f, sr=SR, mono=True)
    m = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    ), ref=np.max).T.astype(np.float32)
    return m[:FRAMES] if len(m)>=FRAMES else np.pad(
        m, ((0,FRAMES-len(m)),(0,0)), constant_values=m.min()
    )
def run(sess, x):
    return sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
def main():
    if len(sys.argv) != 4:
        print("Usage: verify_quant_final.py fp32.onnx qdq.onnx wav_dir")
        sys.exit(1)
    fp32, qdq, wav_dir = sys.argv[1:]
    s_fp32 = ort.InferenceSession(fp32)
    s_qdq  = ort.InferenceSession(qdq)
    files = [os.path.join(wav_dir,f) for f in os.listdir(wav_dir) if f.endswith(".wav")]
    if len(files) < 10:
        raise RuntimeError("Need at least 10 samples (not 1).")
    fp32_out, qdq_out = [], []
    for f in files:
        x = mel(f)[None]
        fp32_out.append(run(s_fp32, x))
        qdq_out.append(run(s_qdq, x))
    fp32_out = np.array(fp32_out)
    qdq_out  = np.array(qdq_out)
    failed = False
    for i, name in enumerate(["valence","arousal"]):
        corr,_ = pearsonr(fp32_out[:,i], qdq_out[:,i])
        diff = np.abs(fp32_out[:,i] - qdq_out[:,i])
        print(name)
        print("  Pearson:", corr)
        print("  Max diff:", diff.max())
        print("  Mean diff:", diff.mean())
        if math.isnan(corr) or corr < 0.92:
            print(f"  [FAIL] {name} Pearson {corr:.4f} < 0.92 threshold")
            failed = True
    if failed:
        sys.exit(1)
    print("PASS: quantization acceptable (Pearson >= 0.92 for all outputs)")
if __name__ == "__main__":
    main()
# @path: quantize_model.py
import sys
import os
import numpy as np
import onnx
import onnxruntime as ort
import librosa
from onnxruntime.quantization import quantize_dynamic, QuantType
SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187
def quantize_model(fp32_path, qdq_path):
    quantize_dynamic(
        fp32_path,
        qdq_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        reduce_range=True,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    print("Saved:", qdq_path)
def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py model_fp32.onnx model_qdq.onnx")
        sys.exit(1)
    fp32_path = sys.argv[1]
    qdq_path  = sys.argv[2]
    global mean, std
    if os.path.exists("emb_mean.npy") and os.path.exists("emb_std.npy"):
        mean = np.load("emb_mean.npy").astype(np.float32)
        std  = np.load("emb_std.npy").astype(np.float32)
    
    quantize_model(fp32_path, qdq_path)
    wav_dir = "test_wavs"
    if not os.path.isdir(wav_dir):
        print(f"[WARN] Missing '{wav_dir}/' → skipping validation")
        return
    files = list_wavs(wav_dir)
    if not files:
        print(f"[WARN] No WAV files in '{wav_dir}/' → skipping validation")
        return
    fp32_sess = load_session(fp32_path)
    qdq_sess  = load_session(qdq_path)
    results_fp32 = []
    results_qdq  = []
    for f in files:
        x = preprocess_audio(f)
        y_fp32 = run(fp32_sess, x)
        y_qdq  = run(qdq_sess, x)
        results_fp32.append(y_fp32)
        results_qdq.append(y_qdq)
    metrics = compute_metrics(results_fp32, results_qdq)
    print_report(metrics)
def list_wavs(dir_path):
    return [
        os.path.join(dir_path, f)
        for f in sorted(os.listdir(dir_path))
        if f.lower().endswith(".wav")
    ]
def load_session(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
def preprocess_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_t = mel_db.T.astype(np.float32)
    if mel_t.shape[0] >= FRAMES:
        mel_fixed = mel_t[:FRAMES]
    else:
        pad = np.full(
            (FRAMES - mel_t.shape[0], N_MELS),
            mel_t.min() if mel_t.size else -80.0,
            dtype=np.float32
        )
        mel_fixed = np.vstack((mel_t, pad))
    return mel_fixed[None, :, :]
def run(session, x):
    inp = session.get_inputs()[0]
    name = inp.name
    if len(inp.shape) >= 3 and inp.shape[2] == x.shape[1]:
        x = x.transpose(0, 2, 1)
    out = session.run(None, {name: x.astype(np.float32)})[0]
    if "mean" in globals() and "std" in globals():
        if out.shape[-1] == mean.shape[0]:
            out = (out - mean) / np.maximum(std, 1e-8)
    return out
def compute_metrics(A, B):
    A = np.vstack(A)
    B = np.vstack(B)
    diff = np.abs(A - B)
    max_diff  = diff.max()
    mean_diff = diff.mean()
    corr_0 = pearson(A[:, 0], B[:, 0])
    corr_1 = pearson(A[:, 1], B[:, 1])
    return {
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "pearson_valence": float(corr_0),
        "pearson_arousal": float(corr_1),
        "n_samples": len(A),
    }
def pearson(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    if denom < 1e-12:
        return 0.0
    return float((xm * ym).sum() / denom)
def print_report(m):
    print("Max diff:", m["max_diff"])
    print("Mean diff:", m["mean_diff"])
    corr_v = m["pearson_valence"]
    corr_a = m["pearson_arousal"]
    n      = m.get("n_samples", "?")
    print("Corr valence:", corr_v)
    print("Corr arousal:", corr_a)
    if np.isnan(corr_v) or np.isnan(corr_a):
        print(f"[WARN] Pearson undefined (n_samples={n}); using diff-based check.")
        if m["max_diff"] <= 0.01 and m["mean_diff"] <= 0.001:
            print("PASS: quantization acceptable (diff-based)")
        else:
            print("FAIL: quantization degraded model (diff-based)")
    elif corr_v < 0.92 or corr_a < 0.92:
        print("FAIL: quantization degraded model")
    else:
        print("PASS: quantization acceptable")
if __name__ == "__main__":
    main()
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
