# @path: run_inference.py
import argparse

import numpy as np

from audio_utils import build_model_input_from_path, get_cpu_session

def apply_activation(output, mode):
    output = np.asarray(output, dtype=np.float32).reshape(-1)
    if output.size != 2:
        raise RuntimeError(f"Expected 2 output values, got shape {output.shape}")
    val, aro = output

    if mode == "tanh":
        return np.tanh(val), np.tanh(aro)

    if mode == "sigmoid":
        return (
            1.0 / (1.0 + np.exp(-val)),
            1.0 / (1.0 + np.exp(-aro)),
        )

    if mode == "raw":
        return val, aro

    raise ValueError("Invalid activation mode")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--activation", default="raw", choices=["raw", "tanh", "sigmoid"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Loading audio:", args.audio)

    sess = get_cpu_session(args.model)
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    model_input = build_model_input_from_path(args.audio, input_meta.shape)

    out = sess.run(None, {input_name: model_input.astype(np.float32)})[0]
    raw_output = np.asarray(out).squeeze().astype(np.float32)
    if raw_output.size != 2:
        raise RuntimeError(f"Expected merged model to return 2 values, got shape {raw_output.shape}")

    print("Raw model output:", raw_output.tolist())

    valence, arousal = apply_activation(raw_output, args.activation)

    print("Final Valence:", float(valence))
    print("Final Arousal:", float(arousal))

if __name__ == "__main__":
    main()
