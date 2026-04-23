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
