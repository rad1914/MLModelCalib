# @path: merging.py
import sys, onnx, numpy as np
from onnx import helper, numpy_helper as nh
from onnx.compose import merge_models, add_prefix
a = sys.argv
len(a) < 6 and sys.exit("Usage: merge_and_stitch.py ENC HEAD MEAN STD OUT [--prefix p] [--tanh]")
p = a[a.index("--prefix")+1] if "--prefix" in a else "h_"
enc = onnx.load(a[1])
head = add_prefix(onnx.load(a[2]), p)
merged = merge_models(enc, head, io_map=[(enc.graph.output[0].name, head.graph.input[0].name)])
g = merged.graph
eo = enc.graph.output[0].name
hi = head.graph.input[0].name
g.initializer.extend([
    nh.from_array(np.load(a[3]).astype(np.float32), "mean"),
    nh.from_array(np.load(a[4]).astype(np.float32), "std"),
])
sub_out = eo + "_sub"
div_out = eo + "_div"
idx = next((i for i, n in enumerate(g.node) if hi in n.input), len(g.node))
g.node.insert(idx, helper.make_node("Sub", [eo, "mean"], [sub_out]))
g.node.insert(idx + 1, helper.make_node("Div", [sub_out, "std"], [div_out]))
for n in g.node:
    if any(o in (sub_out, div_out) for o in n.output):
        continue
    n.input[:] = [div_out if i == hi else i for i in n.input]
if "--tanh" in a:
    o = g.output[0].name
    t = o+"_t"
    g.node.append(helper.make_node("Tanh", [o], [t]))
    g.output[0].name = t
    for v in g.value_info:
        if v.name == o: v.name = t
g.ClearField("value_info")
onnx.save(merged, a[5])
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
SR,N_FFT,HOP,N_MELS,FRAMES=16000,512,256,96,187
def mel(p):
    y,_=librosa.load(p,sr=SR,mono=True)
    m=librosa.power_to_db(
        librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS),
        ref=np.max
    ).T.astype(np.float32)
    return m[:FRAMES] if len(m)>=FRAMES else np.pad(
        m,((0,FRAMES-len(m)),(0,0)),'constant',
        constant_values=(m.min() if len(m) else -80)
    )
def run(s,x): return s.run(None,{s.get_inputs()[0].name:x})[0]
def run1(p,x):
    s=ort.InferenceSession(p,providers=["CPUExecutionProvider"])
    return run(s,x[None])[0]
def add_dbg(i,o):
    m=onnx.load(i); n=m.graph.node
    t=[x for x in n if x.op_type=="Tanh"]
    hr=(t[-1].input[0] if t else n[-1].output[0])
    outs={y for x in n for y in x.output}
    es=next((x for x in ("emb_stdized","emb_sub","emb_std","head_input") if x in outs),None)
    ex={x.name for x in m.graph.output}
    for x in (es,hr):
        if x and x not in ex:
            m.graph.output.append(helper.make_tensor_value_info(x,TensorProto.FLOAT,None))
    onnx.save(m,o)
    return es,hr
def ref(x,e,h,m,s):
    e,h=ort.InferenceSession(e),ort.InferenceSession(h)
    z=(run(e,x)[0]-m)/np.maximum(s,1e-8)
    r=run(h,z.astype(np.float32))[0]
    return z,r,np.tanh(r)
def run_all(p,x):
    s=ort.InferenceSession(p)
    o=s.run(None,{s.get_inputs()[0].name:x})
    return dict(zip([i.name for i in s.get_outputs()],o))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--audio",default="test.wav")
    p.add_argument("--enc",default="msd_musicnn.onnx")
    p.add_argument("--head",default="deam_head.onnx")
    p.add_argument("--merged",required=True)
    p.add_argument("--debug_out",default="merged_debug.onnx")
    p.add_argument("--mean",default="emb_mean.npy")
    p.add_argument("--std",default="emb_std.npy")
    a=p.parse_args()
    M,S=np.load(a.mean),np.load(a.std)
    x=mel(a.audio)[None]
    z,r,py=ref(x,a.enc,a.head,M,S)
    mg=run(ort.InferenceSession(a.merged),x)[0]
    d=np.abs(py-mg)
    print("Python:",py)
    print("Merged:",mg)
    print("Diff:",d,"Max:",d.max())
    es,hr=add_dbg(a.merged,a.debug_out)
    r=run_all(a.debug_out,x)
    print("\nDebug:",list(r))
    if es in r: print("z:",r[es].mean(),r[es].std())
    if hr in r: print("raw:",r[hr].ravel()[:10])
    if "final_output" in r: print("final:",r["final_output"])
    print("\nMerged-only:",run1(a.merged,mel(a.audio)))
if __name__=="__main__": main()
# @path: gen_head_calib.py
import sys, glob, os, numpy as np, librosa, onnxruntime as ort
SR,N_FFT,HOP,N_MELS,FRAMES = 16000,512,256,96,187
def mel(f):
    y,_ = librosa.load(f, sr=SR, mono=True)
    m = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    ), ref=np.max).T.astype('float32')
    return m[:FRAMES] if len(m)>=FRAMES else np.vstack([m, np.full((FRAMES-len(m),N_MELS), m.min() if len(m) else -80, 'float32')])
if len(sys.argv)!=5: sys.exit("Usage: gen_head_calib.py ENC DIR MEAN STD")
enc,d,mp,sp = sys.argv[1:]
os.makedirs("head_calib", exist_ok=True)
mean,std = np.load(mp).astype('float32'), np.load(sp).astype('float32')
std[std==0]=1
sess = ort.InferenceSession(enc, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
for i,f in enumerate(glob.glob(d+"/*.wav")):
    e = sess.run(None, {inp: mel(f)[None]})[0].squeeze().astype('float32')
    np.save(f"head_calib/calib_{i:04d}.npy", (e-mean)/std)
print("Saved", i+1 if 'i' in locals() else 0)
# @path: audio_utils.py
import numpy as np, librosa
def load_audio_mono(p, sr=16000):
    return librosa.load(p, sr=sr, mono=True)[0].astype(np.float32)
def make_mel_patch(y, sr=16000, n_fft=512, hop=256, n_mels=96, frames=187, power=2.0):
    m = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=power),
        ref=np.max
    ).T.astype(np.float32)
    return m[:frames] if m.shape[0] >= frames else np.vstack([m, np.full((frames - m.shape[0], n_mels), m.min() if m.size else -80.0, np.float32)])
def prepare_model_input(p, s):
    a = p[None].astype(np.float32)
    if not s: return a
    s = list(s)
    ok = lambda x,y,i,j: (s[i] is None or int(s[i])==x) and (s[j] is None or int(s[j])==y)
    return (
        a if len(s)==3 and ok(a.shape[1],a.shape[2],1,2) else
        a.transpose(0,2,1) if len(s)==3 else
        a[...,None] if len(s)==4 and ok(a.shape[1],a.shape[2],1,2) else
        a[:,None] if len(s)==4 and ok(a.shape[1],a.shape[2],2,3) else
        a.transpose(0,2,1)[...,None] if len(s)==4 and ok(a.shape[2],a.shape[1],1,2) else
        a
    )
def standardize_embedding(e, m, s, eps=1e-6):
    e,m,s = map(lambda x: np.asarray(x,np.float32).ravel(), (e,m,s))
    if m.size==1: m = np.full_like(e,m[0])
    if s.size==1: s = np.full_like(e,s[0])
    if m.size!=e.size or s.size!=e.size: raise ValueError("size mismatch")
    return ((e-m)/np.where(np.abs(s)<eps,1.0,s)).astype(np.float32)
# @path: inject_standardizer.py
import argparse, os, sys, numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto
p = argparse.ArgumentParser()
p.add_argument('-m', required=True)
p.add_argument('--mean', required=True)
p.add_argument('--std', required=True)
p.add_argument('-o', required=True)
p.add_argument('--head')
a = p.parse_args()
if not (os.path.isfile(a.m) and os.path.isfile(a.mean) and os.path.isfile(a.std)):
    sys.exit("missing file")
m = onnx.load(a.m)
g = m.graph
nodes = list(g.node)
head = a.head or next((n.input[0] for n in g.node if n.op_type.lower()=='tanh' and n.input), None)
if not head: sys.exit("no head")
mean = np.load(a.mean).astype(np.float32).ravel()
std  = np.load(a.std ).astype(np.float32).ravel()
if mean.shape != std.shape: sys.exit("shape mismatch")
g.initializer += [
    numpy_helper.from_array(mean, "mean"),
    numpy_helper.from_array(std,  "std")
]
sub, div = "sub", "out"
nodes.insert(
    next((i for i,n in enumerate(nodes) if head in n.input), len(nodes)),
    helper.make_node('Sub', [head, "mean"], [sub])
)
nodes.insert(
    next((i for i,n in enumerate(nodes) if sub in n.input), len(nodes)),
    helper.make_node('Div', [sub, "std"], [div])
)
for n in nodes:
    n.input[:] = [div if i==head else i for i in n.input]
for o in g.output:
    if o.name == head: o.name = div
g.node[:] = nodes
g.value_info += [
    helper.make_tensor_value_info(sub, TensorProto.FLOAT, ['N', mean.size]),
    helper.make_tensor_value_info(div, TensorProto.FLOAT, ['N', mean.size])
]
onnx.save(m, a.o)
#!/usr/bin/env python3
# @path: runners.py
import os, sys, json, argparse
import numpy as np, soundfile as sf, librosa, onnxruntime as ort
SR,N_FFT,HOP,N_MELS,FRAMES=16000,512,256,96,187
act={"raw":lambda x:x,"tanh":np.tanh,"sigmoid":lambda x:1/(1+np.exp(-x))}
die=lambda p: os.path.exists(p) or sys.exit(f"Missing: {p}")
def wav(p,sr):
    die(p); y,s=sf.read(p,dtype='float32')
    if y.ndim>1: y=y.mean(1)
    return librosa.resample(y, orig_sr=s, target_sr=sr) if s!=sr else y
def mel(y,sr,f):
    m=librosa.power_to_db(librosa.feature.melspectrogram(
        y=y,sr=sr,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS
    ),ref=np.max).T.astype(np.float32)
    return m[:f] if len(m)>=f else np.vstack([m,np.full((f-len(m),N_MELS),m.min() if len(m) else -80,np.float32)])
load=lambda p:(np.load(p) if p.endswith('.npy') else np.fromfile(p,np.float32)).astype(np.float32)
def run(s,x):
    i=s.get_inputs()[0]
    x=x.T[None] if len(i.shape)>=3 and i.shape[2]==x.shape[0] else x[None]
    return s.run(None,{i.name:x.astype(np.float32)})[0]
p=argparse.ArgumentParser()
[p.add_argument(f"--{k}",required=True) for k in ("audio","encoder","head","mean","std")]
p.add_argument("--sr",type=int,default=SR)
p.add_argument("--frames",type=int,default=FRAMES)
p.add_argument("--activation",default="tanh",choices=act)
a=p.parse_args()
[die(getattr(a,k)) for k in ("audio","encoder","head","mean","std")]
y=wav(a.audio,a.sr)
m=mel(y,a.sr,a.frames)
emb=run(ort.InferenceSession(a.encoder,providers=["CPUExecutionProvider"]),m).ravel()
mean,std=load(a.mean),load(a.std)
if mean.size==1: mean=np.full_like(emb,mean.item())
if std.size==1: std=np.full_like(emb,std.item())
if mean.size!=emb.size or std.size!=emb.size: sys.exit("mean/std size mismatch")
emb=(emb-mean)/(std+1e-12)
out=run(ort.InferenceSession(a.head,providers=["CPUExecutionProvider"]),emb.reshape(1,-1))
print(json.dumps({"output":act[a.activation](out)[0].tolist()}))
#!/usr/bin/env python3
# @path: quantize_model.py
import argparse
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_in")
    p.add_argument("model_out")
    p.add_argument("calib_dir")
    args = p.parse_args()
    quantize_dynamic(
        args.model_in,
        args.model_out,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    print("Saved:", args.model_out)
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# @path: compute_calib_stats.py
import os, sys, argparse, numpy as np, onnxruntime as ort
from audio_utils import load_audio_mono, make_mel_patch
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", default="msd_musicnn.onnx")
    p.add_argument("-c", "--calib", default="calib_wavs")
    p.add_argument("--out-mean", default="emb_mean.npy")
    p.add_argument("--out-std", default="emb_std.npy")
    a = p.parse_args()
    if not os.path.isfile(a.model):
        sys.exit(f"Missing model: {a.model}")
    if not os.path.isdir(a.calib):
        sys.exit(f"Missing calib dir: {a.calib}")
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = inp.shape
    name = inp.name
    def prep(x):
        x = x.astype(np.float32)
        if len(shape) == 4:
            return x[None, None]
        return x[None]
    files = [f for f in sorted(os.listdir(a.calib)) if f.lower().endswith(".wav")]
    total = len(files)
    if total == 0:
        sys.exit("No WAV files found")
    E = []
    done = 0
    for i, f in enumerate(files, 1):
        pth = os.path.join(a.calib, f)
        try:
            y = load_audio_mono(pth)
            x = make_mel_patch(y)
            out = sess.run(None, {name: prep(x)})[0]
            E.append(out.reshape(-1))
            done += 1
        except Exception as e:
            print(f"[skip] {pth}: {e}", file=sys.stderr)
        print(f"\r[{i}/{total}] processed={done}", end="", flush=True)
    print()
    if not E:
        sys.exit("No valid WAVs processed")
    E = np.stack(E).astype(np.float32)
    std = E.std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    np.save(a.out_mean, E.mean(0).astype(np.float32))
    np.save(a.out_std, std)
    print(f"Saved: {a.out_mean}")
    print(f"Saved: {a.out_std}")
if __name__ == "__main__":
    main()
