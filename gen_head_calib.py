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
