# @path: quantize_head.py
import sys, glob, os, numpy as np, librosa, onnxruntime as ort
from onnxruntime.quantization import *
M,O,D=sys.argv[1:4]
SR,NF,H,NM,F=16000,512,256,96,187
I=ort.InferenceSession(M,providers=["CPUExecutionProvider"]).get_inputs()[0].name
def m(p):
    y,_=librosa.load(p,sr=SR,mono=True)
    x=librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=SR,n_fft=NF,hop_length=H,n_mels=NM),ref=np.max).T.astype(np.float32)
    return np.vstack([x,np.full((F-len(x),NM),x.min() if len(x) else -80,np.float32)])[:F]
class R(CalibrationDataReader):
    def __init__(s,f):s.f=f;s.i=0
    def get_next(s):
        if s.i>=len(s.f):return
        x=m(s.f[s.i]);s.i+=1
        return {I:x[None].astype(np.float32)}
f=sorted(glob.glob(os.path.join(D,"*.wav")))
if not f:raise RuntimeError("No WAVs")
quantize_static(M,O,R(f),quant_format=QuantFormat.QDQ,per_channel=False,activation_type=QuantType.QInt8,weight_type=QuantType.QInt8)
