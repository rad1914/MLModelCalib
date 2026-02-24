#!/usr/bin/env python3

import argparse
import numpy as np
import onnxruntime as ort
import librosa


SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187


def make_mel(path):
    y, _ = librosa.load(path, sr=SR, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)

    if mel_db.shape[0] >= FRAMES:
        return mel_db[:FRAMES]

    pad = FRAMES - mel_db.shape[0]
    pad_val = mel_db.min() if mel_db.shape[0] > 0 else -80.0

    return np.vstack(
        [mel_db, np.full((pad, N_MELS), pad_val, dtype=np.float32)]
    )


def run_onnx(model_path, mel):
    sess = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    input_name = sess.get_inputs()[0].name

    outputs = sess.run(
        None,
        {input_name: mel[np.newaxis].astype(np.float32)}
    )

    return outputs[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)

    args = parser.parse_args()

    mel = make_mel(args.audio)

    out = run_onnx(args.model, mel)

    print("Model:", args.model)
    print("Output:", out)


if __name__ == "__main__":
    main()