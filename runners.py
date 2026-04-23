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
