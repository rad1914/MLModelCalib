#!/usr/bin/env python3
import onnx
import numpy as np
import onnxruntime as ort
import librosa
from onnx import helper, TensorProto, numpy_helper

# Paths (adjust if needed)
AUDIO = "test.wav"
ENCODER_ONNX = "msd_musicnn.onnx"
HEAD_ONNX = "deam_head.onnx"
MERGED = "merged_float.onnx"
EMB_MEAN = "emb_mean.npy"
EMB_STD = "emb_std.npy"
DEBUG_OUT = "merged_debug.onnx"

# audio/mel params (must match training)
SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187

def make_mel(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)
    T = mel_db.shape[0]
    if T >= FRAMES:
        mel_db = mel_db[:FRAMES, :]
    else:
        pad_rows = FRAMES - T
        pad_value = mel_db.min() if T > 0 else -80.0
        mel_db = np.vstack([mel_db, np.full((pad_rows, N_MELS), pad_value, dtype=np.float32)])
    return mel_db

def describe_graph(model_path):
    m = onnx.load(model_path)
    print(f"\n=== Graph description: {model_path} ===")
    print("Inputs:")
    for i in m.graph.input:
        print(" ", i.name)
    print("Outputs:")
    for o in m.graph.output:
        print(" ", o.name)
    # find head input / head output candidates by heuristic (contains 'head' or 'Identity' etc)
    names = [n.name for n in m.graph.node]
    print("\nNodes summary (first 50):")
    for n in m.graph.node[:50]:
        print(" -", n.name, "op:", n.op_type, "inputs:", list(n.input)[:3], "-> outputs:", list(n.output)[:3])
    return m

def add_debug_outputs(orig_path, debug_path, emb_stdized_name_guess=None, head_output_name_guess=None):
    m = onnx.load(orig_path)

    # Heuristics: if user provided guesses, try those, else try to find plausible names.
    # We'll attempt to find a node with op_type 'Tanh' and use its input name as 'head_raw'
    tanh_nodes = [n for n in m.graph.node if n.op_type == "Tanh"]
    head_raw_name = None
    if tanh_nodes:
        # assume last Tanh is our final, its input is head_raw
        head_raw_name = tanh_nodes[-1].input[0]
        print("Detected Tanh node; using its input as head_raw:", head_raw_name)
    else:
        # fallback: look for a node producing a 2-dim output (name containing 'Identity' or 'output')
        for n in m.graph.node:
            if n.op_type in ("Identity", "Gemm", "MatMul", "Add"):
                for out in n.output:
                    # crude: hope the head output name exists with shape 1x2 in value_info (not guaranteed)
                    head_raw_name = out
                    break
            if head_raw_name:
                break

    emb_std_name = None
    # Try to find 'emb_stdized' name we inserted earlier
    candidate_names = set()
    for n in m.graph.node:
        for out in n.output:
            candidate_names.add(out)
    for guess in ("emb_stdized", "emb_sub", "emb_sub:0", "emb_std", "head_input"):
        if guess in candidate_names:
            emb_std_name = guess
            break
    # If not found, try nodes that have shape ~ (1,200) by searching initializer sizes:
    if emb_std_name is None:
        # try to find a node whose output is consumed by many nodes (likely embedding)
        # fallback: print candidates to user and abort adding debug outputs
        print("Could not auto-detect emb_stdized tensor name; available node outputs (sample 50):")
        sample = list(candidate_names)[:50]
        for s in sample:
            print(" ", s)
        raise SystemExit("Provide emb_stdized name manually if auto detection failed.")

    # Add outputs for emb_stdized and head_raw_name
    # Create ValueInfoProtos assuming float and unknown shape (let runtime infer)
    vi1 = helper.make_tensor_value_info(emb_std_name, TensorProto.FLOAT, None)
    vi2 = helper.make_tensor_value_info(head_raw_name, TensorProto.FLOAT, None)

    # To avoid duplicating outputs with same name, check existing outputs
    existing_out_names = [o.name for o in m.graph.output]
    if emb_std_name not in existing_out_names:
        m.graph.output.extend([vi1])
        print("Added debug output:", emb_std_name)
    if head_raw_name not in existing_out_names:
        m.graph.output.extend([vi2])
        print("Added debug output:", head_raw_name)

    onnx.save(m, debug_path)
    print("Saved debug model to", debug_path)
    return debug_path, emb_std_name, head_raw_name

def run_reference_pipeline(mel):
    # encoder -> emb -> stdize -> head -> tanh
    enc_sess = ort.InferenceSession(ENCODER_ONNX, providers=["CPUExecutionProvider"])
    head_sess = ort.InferenceSession(HEAD_ONNX, providers=["CPUExecutionProvider"])

    enc_in = enc_sess.get_inputs()[0].name
    emb = enc_sess.run(None, {enc_in: mel[np.newaxis,:,:].astype(np.float32)})[0]  # (1,200)

    emb_mean = np.load(EMB_MEAN).astype(np.float32)
    emb_std = np.load(EMB_STD).astype(np.float32)
    emb_std = np.where(emb_std < 1e-8, 1.0, emb_std)
    emb_stdized = (emb - emb_mean) / emb_std

    head_in = head_sess.get_inputs()[0].name
    raw = head_sess.run(None, {head_in: emb_stdized.astype(np.float32)})[0][0]
    final = np.tanh(raw)
    return emb.astype(np.float32), emb_stdized.astype(np.float32), raw.astype(np.float32), final.astype(np.float32)

def run_debug_merged(mel, debug_model_path, emb_std_name, head_raw_name):
    sess = ort.InferenceSession(debug_model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outs = sess.run(None, {input_name: mel[np.newaxis,:,:].astype(np.float32)})
    # ONNX runtime returns outputs in order; find indices for our added outputs
    out_names = [o.name for o in sess.get_outputs()]
    result_map = {name: outs[idx] for idx, name in enumerate(out_names)}
    return result_map

if __name__ == "__main__":
    print("Making mel from", AUDIO)
    mel = make_mel(AUDIO)

    print("\nDescribing merged graph before debug modification:")
    describe_graph(MERGED)

    print("\nAdding debug outputs to merged model (emb_stdized and head_raw)...")
    debug_path, emb_std_name, head_raw_name = add_debug_outputs(MERGED, DEBUG_OUT)

    print("\nRunning python reference pipeline...")
    emb, emb_stdized_py, head_raw_py, final_py = run_reference_pipeline(mel)
    print("Python encoder embedding mean/std:", float(emb.mean()), float(emb.std()))
    print("Python emb_stdized mean/std:", float(emb_stdized_py.mean()), float(emb_stdized_py.std()))
    print("Python head_raw:", head_raw_py)
    print("Python final (tanh):", final_py)

    print("\nRunning merged debug model...")
    merged_results = run_debug_merged(mel, debug_path, emb_std_name, head_raw_name)
    print("Merged debug outputs keys:", list(merged_results.keys()))

    # Try to print our two debug tensors if available
    if emb_std_name in merged_results:
        em = merged_results[emb_std_name]
        print("Merged emb_stdized mean/std:", float(em.mean()), float(em.std()))
    else:
        print("Merged emb_stdized not present in outputs.")

    if head_raw_name in merged_results:
        hr = merged_results[head_raw_name]
        print("Merged head_raw:", np.array(hr).reshape(-1)[:10])  # print first values
    else:
        print("Merged head_raw not present in outputs.")

    # Also print final_output if present
    if "final_output" in merged_results:
        print("Merged final_output:", merged_results["final_output"])
    else:
        print("Merged final_output not present in debug outputs.")