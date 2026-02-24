#!/usr/bin/env python3
import argparse
import numpy as np
import librosa
import onnxruntime as ort

# =====================
# CONSTANTS (MUST MATCH TRAINING)
# =====================
SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187  # ~3 seconds


# =====================
# AUDIO
# =====================
def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def make_mel_patch(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db_t = mel_db.T.astype(np.float32)

    T = mel_db_t.shape[0]

    if T >= FRAMES:
        patch = mel_db_t[:FRAMES, :]
    else:
        pad_rows = FRAMES - T
        pad_value = mel_db_t.min() if T > 0 else -80.0
        patch = np.vstack([
            mel_db_t,
            np.full((pad_rows, N_MELS), pad_value, dtype=np.float32)
        ])

    return patch


# =====================
# ACTIVATIONS
# =====================
def apply_activation(output, mode):
    val, aro = output

    if mode == "tanh":
        return np.tanh(val), np.tanh(aro)

    elif mode == "sigmoid":
        return (
            1.0 / (1.0 + np.exp(-val)),
            1.0 / (1.0 + np.exp(-aro))
        )

    elif mode == "raw":
        return val, aro

    else:
        raise ValueError("Invalid activation mode")


# =====================
# MAIN
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--emb-mean", required=True)
    parser.add_argument("--emb-std", required=True)
    parser.add_argument("--activation", default="tanh",
                        choices=["raw", "tanh", "sigmoid"])
    args = parser.parse_args()

    print("Loading audio:", args.audio)
    y = load_audio(args.audio)

    mel_patch = make_mel_patch(y)
    print("Mel patch shape:", mel_patch.shape)

    # =====================
    # ENCODER
    # =====================
    enc_sess = ort.InferenceSession(
        args.encoder,
        providers=["CPUExecutionProvider"]
    )

    enc_input_name = enc_sess.get_inputs()[0].name

    emb = enc_sess.run(
        None,
        {enc_input_name: mel_patch[np.newaxis, :, :].astype(np.float32)}
    )[0]

    print("Raw embedding shape:", emb.shape)
    print("Raw embedding mean:", emb.mean())
    print("Raw embedding std:", emb.std())

    # =====================
    # STANDARDIZE
    # =====================
    emb_mean = np.load(args.emb_mean)
    emb_std = np.load(args.emb_std)

    emb_std = np.where(emb_std < 1e-8, 1.0, emb_std)

    emb_stdized = (emb - emb_mean) / emb_std

    print("Standardized embedding mean:", emb_stdized.mean())
    print("Standardized embedding std:", emb_stdized.std())

    # =====================
    # HEAD
    # =====================
    head_sess = ort.InferenceSession(
        args.head,
        providers=["CPUExecutionProvider"]
    )

    head_input_name = head_sess.get_inputs()[0].name

    raw_output = head_sess.run(
        None,
        {head_input_name: emb_stdized.astype(np.float32)}
    )[0][0]

    print("Raw head output:", raw_output)

    # =====================
    # FINAL OUTPUT
    # =====================
    valence, arousal = apply_activation(raw_output, args.activation)

    print("Final Valence:", float(valence))
    print("Final Arousal:", float(arousal))


if __name__ == "__main__":
    main()