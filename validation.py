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
