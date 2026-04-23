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
