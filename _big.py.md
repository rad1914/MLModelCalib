# @path: compute_calib_stats.py

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import librosa
import onnxruntime as ort
from typing import Tuple, List

DEFAULT_SR = 16000
DEFAULT_N_FFT = 512
DEFAULT_HOP = 256
DEFAULT_N_MELS = 96
DEFAULT_FRAMES = 187
DEFAULT_WIN_SEC = 3.0
DEFAULT_HOP_SEC = 1.0
EPS_STD_FLOOR = 1e-6
STD_FLOOR = 1e-6

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute embedding mean/std from WAV calibration set")
    p.add_argument('--model', '-m', default='msd_musicnn.onnx', help='Path to encoder ONNX model')
    p.add_argument('--calib', '-c', default='calib_wavs', help='Directory containing WAV calibration files')
    p.add_argument('--out-mean', default='emb_mean.npy', help='Output .npy file for mean')
    p.add_argument('--out-std', default='emb_std.npy', help='Output .npy file for std')
    p.add_argument('--sr', type=int, default=DEFAULT_SR, help='Audio sampling rate (Hz)')
    p.add_argument('--n-fft', type=int, default=DEFAULT_N_FFT, help='STFT n_fft')
    p.add_argument('--hop', type=int, default=DEFAULT_HOP, help='STFT hop_length (samples)')
    p.add_argument('--n-mels', type=int, default=DEFAULT_N_MELS, help='Number of mel bins')
    p.add_argument('--frames', type=int, default=DEFAULT_FRAMES, help='Number of frames (time axis) expected by the model')
    p.add_argument('--win-sec', type=float, default=DEFAULT_WIN_SEC, help='Window length in seconds (patch size)')
    p.add_argument('--hop-sec', type=float, default=DEFAULT_HOP_SEC, help='Hop/stride between patches in seconds')
    p.add_argument('--power', type=float, default=2.0, help='Power for mel spectrogram (2.0 for power)')
    p.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    return p.parse_args()

def list_wavs(calib_dir: str) -> List[str]:
    files = sorted([f for f in os.listdir(calib_dir) if f.lower().endswith('.wav')])
    return files

def load_audio(path: str, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def make_mel_patch(y: np.ndarray, sr: int, n_fft: int, hop: int, n_mels: int, frames: int, power: float) -> np.ndarray:
    y = y.astype(np.float32)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop,
        n_mels=n_mels, power=power
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, -80.0, 0.0)

    frames_raw = mel_db.shape[1]
    if frames_raw % 2 != 0:
        mel_db = mel_db[:, :frames_raw - 1]

    mel_ds = 0.5 * (mel_db[:, 0::2] + mel_db[:, 1::2])
    mel_db = mel_ds.T

    if mel_db.shape[0] >= frames:
        start = (mel_db.shape[0] - frames)
        patch = mel_db[start:start+frames].astype(np.float32)
    else:
        pad_val = float(mel_db.min()) if mel_db.size else -80.0
        pad = np.full((frames - mel_db.shape[0], n_mels), pad_val, dtype=np.float32)
        patch = np.vstack([mel_db.astype(np.float32), pad])
    return patch

def prepare_input_for_model(patch: np.ndarray, model_input_shape, frames: int, n_mels: int) -> np.ndarray:
    arr = patch[np.newaxis, :, :].astype(np.float32)

    if not model_input_shape:
        return arr

    shape = list(model_input_shape)
    if len(shape) == 3:
        s1 = shape[1]
        s2 = shape[2]
        if (s1 is None or int(s1) == frames) and (s2 is None or int(s2) == n_mels):
            return arr
        if (s1 is None or int(s1) == n_mels) and (s2 is None or int(s2) == frames):
            return arr.transpose(0,2,1)

    if len(shape) == 4:
        if (shape[1] is None or int(shape[1]) == frames) and (shape[2] is None or int(shape[2]) == n_mels):
            return arr[:, :, :, np.newaxis]
        if (shape[2] is None or int(shape[2]) == frames) and (shape[3] is None or int(shape[3]) == n_mels):
            return arr[:, np.newaxis, :, :]
        if (shape[1] is None or int(shape[1]) == n_mels) and (shape[2] is None or int(shape[2]) == frames):
            return arr.transpose(0,2,1)[:, :, :, np.newaxis]

    return arr

def open_session(model_path: str) -> ort.InferenceSession:
    try:
        return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print("Error: Failed to open ONNX model:", e, file=sys.stderr)
        raise

def compute_stats_from_embeddings(emb_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    embs = np.stack(emb_list, axis=0)
    mean = embs.mean(axis=0)
    std = embs.std(axis=0)
    std = np.maximum(std, STD_FLOOR)
    return mean.astype(np.float32), std.astype(np.float32), int(embs.shape[0])

def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(args.calib):
        print(f"ERROR: calib dir not found: {args.calib}", file=sys.stderr)
        sys.exit(2)

    wavs = list_wavs(args.calib)
    if not wavs:
        print("ERROR: No WAV files found in calib directory. Please add representative audio.", file=sys.stderr)
        sys.exit(3)

    print(f"Loading ONNX model: {args.model}")
    sess = open_session(args.model)
    input_meta = sess.get_inputs()[0]
    in_name = input_meta.name
    in_shape = input_meta.shape
    if args.verbose:
        print("Model input name:", in_name, "shape:", in_shape)

    emb_list = []
    total_patches = 0
    win_samples = int(args.win_sec * args.sr)
    hop_samples = int(args.hop_sec * args.sr)
    for i, fname in enumerate(wavs, 1):
        path = os.path.join(args.calib, fname)
        try:
            y = load_audio(path, sr=args.sr)
        except Exception as e:
            print(f"Warning: failed to load {fname}: {e}", file=sys.stderr)
            continue

        if len(y) < win_samples:
            starts = [0]
        else:
            starts = list(range(0, max(1, len(y) - win_samples + 1), hop_samples))

        for s in starts:
            chunk = y[s:s+win_samples]
            patch = make_mel_patch(chunk, sr=args.sr, n_fft=args.n_fft, hop=args.hop,
                                   n_mels=args.n_mels, frames=args.frames, power=args.power)
            inp = prepare_input_for_model(patch, in_shape, frames=args.frames, n_mels=args.n_mels)
            inp = inp.astype(np.float32)
            try:
                out = sess.run(None, {in_name: inp})
            except Exception as e:
                print(f"Warning: model inference failed on {fname} (start={s}): {e}", file=sys.stderr)
                continue
            if not out or len(out[0].shape) == 0:
                print(f"Warning: unexpected encoder output shape for {fname}", file=sys.stderr)
                continue
            emb = np.asarray(out[0]).reshape(-1)

            if emb.shape[0] != 200:
                raise RuntimeError(f"Unexpected embedding dim {emb.shape[0]}")

            emb_list.append(emb.astype(np.float32))
            total_patches += 1

        if args.verbose:
            print(f"[{i}/{len(wavs)}] {fname}: patches={len(starts)} total_patches={total_patches}")

    if not emb_list:
        print("ERROR: No embeddings were produced. Check model input shape and calibration audio.", file=sys.stderr)
        sys.exit(4)

    np.random.shuffle(emb_list)

    mean, std, n_samples = compute_stats_from_embeddings(emb_list)

    np.save(args.out_mean, mean)
    np.save(args.out_std, std)
    print(f"Saved {args.out_mean} and {args.out_std}")
    print(f"Samples (patches): {n_samples}, embedding dim: {mean.shape[0]}")
    print(f"Embedding mean(mean) = {float(mean.mean()):.6f}, embedding mean(std) = {float(std.mean()):.6f}")

if __name__ == '__main__':
    main()
# @path: merge_onnx_with_standardization_fixed.py
import sys
import onnx
from onnx import TensorProto
import numpy as np
from onnx import helper, numpy_helper, ModelProto

def main():
    if len(sys.argv) != 6:
        print("Usage: python merge_onnx_with_standardization_fixed.py encoder.onnx head.onnx emb_mean.npy emb_std.npy out_merged.onnx")
        sys.exit(1)

    enc_path, head_path, mean_path, std_path, out_path = sys.argv[1:]

    enc = onnx.load(enc_path)
    head = onnx.load(head_path)

    if len(enc.graph.output) != 1:
        raise RuntimeError(f"Encoder must have exactly 1 output, got {len(enc.graph.output)}")
    if len(head.graph.input) != 1:
        raise RuntimeError(f"Head must have exactly 1 input, got {len(head.graph.input)}")
    if len(head.graph.output) < 1:
        raise RuntimeError("Head must have at least 1 output")

    enc_out = enc.graph.output[0].name
    head_in  = head.graph.input[0].name

    print("DEBUG:")
    print(" encoder_out:", enc_out)
    print(" head_input :", head_in)

    mean = np.load(mean_path).astype(np.float32)
    std  = np.load(std_path).astype(np.float32)

    if mean.shape != std.shape:
        raise RuntimeError(f"Mean/std shape mismatch: {mean.shape} vs {std.shape}")
    if mean.ndim != 1:
        raise RuntimeError(f"Expected 1D embedding stats, got mean.ndim={mean.ndim}")
    if mean.size != 200:
        raise RuntimeError(f"Expected 200-d embedding stats, got {mean.size}")

    std = np.maximum(std, 1e-6).astype(np.float32)

    merged = ModelProto()
    merged.CopyFrom(enc)

    if enc_out == head_in:
        raise RuntimeError("Encoder output name collides with head input (will silently break graph)")

    mean_const = numpy_helper.from_array(mean.reshape(1, -1), name="std_mean_const")
    std_const  = numpy_helper.from_array(std.reshape(1, -1),  name="std_std_const")

    merged.graph.initializer.extend([mean_const, std_const])

    sub_out = enc_out + "_sub"
    div_out = enc_out + "_stded"

    sub_node = helper.make_node("Sub", [enc_out, "std_mean_const"], [sub_out], name="std_sub")
    div_node = helper.make_node("Div", [sub_out, "std_std_const"],  [div_out], name="std_div")
    merged.graph.node.extend([sub_node, div_node])

    print("DEBUG:")
    print(" standardization:", enc_out, "->", sub_out, "->", div_out)

    prefix = "head_"
    all_names = set()

    for n in head.graph.node:
        all_names.update(n.input)
        all_names.update(n.output)

    for vi in head.graph.value_info:
        all_names.add(vi.name)

    for o in head.graph.output:
        all_names.add(o.name)

    all_names.discard(head_in)

    name_map = {name: prefix + name for name in all_names}

    def map_name(n):
        return name_map.get(n, n)

    if head_in not in [n for node in head.graph.node for n in list(node.input) + list(node.output)]:
        raise RuntimeError(f"Head input '{head_in}' not found in head graph")
    
    for init in head.graph.initializer:
        arr = numpy_helper.to_array(init)
        new_name = map_name(init.name)
        merged.graph.initializer.append(numpy_helper.from_array(arr, name=new_name))

    for node in head.graph.node:
        mapped_inputs = [div_out if inp == head_in else map_name(inp) for inp in node.input]
        mapped_outputs = [map_name(out) for out in node.output]
        new_node = helper.make_node(
            node.op_type,
            mapped_inputs,
            mapped_outputs,
            name=prefix + (node.name or node.op_type)
        )
        for attr in node.attribute:
            new_node.attribute.extend([attr])
        merged.graph.node.append(new_node)

    print("DEBUG:")
    print(" last 5 nodes:", [n.op_type for n in merged.graph.node[-5:]])

    if merged.graph.node[-1].op_type in {"Sub", "Div"}:
        raise RuntimeError("Graph terminated at standardization (head not connected)")

    for vi in head.graph.value_info:
        new_vi = onnx.ValueInfoProto()
        new_vi.CopyFrom(vi)
        new_vi.name = map_name(vi.name)
        merged.graph.value_info.append(new_vi)

    merged.graph.output.clear()
    for o in head.graph.output:
        new_o = onnx.ValueInfoProto()
        new_o.CopyFrom(o)
        new_o.name = map_name(o.name)
        merged.graph.output.append(new_o)

    print("DEBUG:")
    print(" final outputs:", [o.name for o in merged.graph.output])

    if len(merged.graph.output) != len(head.graph.output):
        raise RuntimeError("Failed to propagate head outputs into merged graph")

    try:
        inferred = onnx.shape_inference.infer_shapes(merged)
        onnx.checker.check_model(inferred)
        onnx.save(inferred, out_path)
        print("OK: merged + standardized ONNX saved to", out_path)
    except Exception as e:
        print("VALIDATION FAILED:", e)
        bad = out_path + ".raw.onnx"
        onnx.save(merged, bad)
        print("Saved raw for debugging:", bad)
        raise

if __name__ == "__main__":
    main()
# @path: merged_runner.py
import argparse
import numpy as np
import onnxruntime as ort
import librosa
import sys
from typing import Optional

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187

def make_mel(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)
    if mel_db.shape[0] >= FRAMES:
        return mel_db[:FRAMES]
    pad = FRAMES - mel_db.shape[0]
    pad_val = mel_db.min() if mel_db.shape[0] > 0 else -80.0
    return np.vstack([mel_db, np.full((pad, N_MELS), pad_val, dtype=np.float32)])

def run_onnx(model_path: str, input_arr: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: input_arr.astype(np.float32)})

    out0 = np.array(out[0])
    return out0.squeeze()

def run_encoder_get_emb(encoder_path: str, mel: np.ndarray) -> np.ndarray:
    emb = run_onnx(encoder_path, mel[np.newaxis])
    return emb.squeeze()

def run_head_on_std_emb(head_path: str, std_emb: np.ndarray) -> np.ndarray:
    out = run_onnx(head_path, std_emb[np.newaxis])
    return out.squeeze()

def find_best_slice(big: np.ndarray, small: np.ndarray):
    big1 = big.ravel()
    small1 = small.ravel()
    n, m = big1.size, small1.size
    if m > n:
        return None
    best_i = None
    best_l2 = float("inf")
    for i in range(0, n - m + 1):
        s = big1[i:i+m]
        d = np.linalg.norm(s - small1)
        if d < best_l2:
            best_l2 = d
            best_i = i
    return best_i, best_l2

def safe_load_np(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    return np.load(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--merged", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--mean", required=False)
    p.add_argument("--std", required=False)
    p.add_argument("--head", required=False, help="Path to head ONNX (deam_head.onnx)")
    p.add_argument("--encoder", required=False, help="Optional encoder ONNX (msd_musicnn.onnx) for parity/search")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    mel = make_mel(args.test)
    if args.verbose:
        print("mel.shape:", mel.shape)

    print("Running merged model:", args.merged)
    merged_out = run_onnx(args.merged, mel[np.newaxis])
    merged_out = np.array(merged_out).squeeze()
    print("merged_out shape:", merged_out.shape, "size:", merged_out.size)

    if args.verbose:
        if merged_out.size not in (2, 200):
            print("WARNING: unexpected output size:", merged_out.size)

        print("merged_out dtype:", merged_out.dtype)
        print("merged_out first_10:", merged_out.ravel()[:10].tolist())

    if merged_out.size == 2:
        va = merged_out.ravel()
        print("Merged model produced valence/arousal:", va.tolist())
        return

    mean = safe_load_np(args.mean) if args.mean else None
    std = safe_load_np(args.std) if args.std else None

    if mean is not None and merged_out.size == mean.size:
        if args.verbose:
            print("Merged output length equals mean length -> interpreting as embedding")
            print("mean.shape:", mean.shape, "std.shape:", None if std is None else std.shape)
        if std is None:
            print("ERROR: std missing for embedding interpretation", file=sys.stderr)
            sys.exit(2)

        if mean.shape[0] != 200:
            print("ERROR: unexpected embedding size", mean.shape, file=sys.stderr)
            sys.exit(3)

        emb = merged_out.astype(np.float32).reshape(mean.shape)
        std_safe = np.where(std == 0.0, 1.0, std) if std is not None else np.ones_like(mean)
        std_emb = (emb - mean) / std_safe
        if args.head is None:
            print("HEAD model is required to process standardized embedding. Provide --head path.")
            sys.exit(2)
        head_out = run_head_on_std_emb(args.head, std_emb)
        print("Valence/Arousal from head (merged->std_emb->head):", head_out.tolist())
        return

    if args.head and args.encoder:
        print("Merged output != 2 and != emb_dim. Will compute encoder->head and search for a matching slice inside merged output.")
        enc_emb = run_encoder_get_emb(args.encoder, mel)
        enc_emb = np.array(enc_emb).squeeze()
        if mean is not None:
            mean = mean.reshape(enc_emb.shape)
            std = std.reshape(enc_emb.shape)
            std_safe = np.where(std == 0.0, 1.0, std)
            std_emb = (enc_emb - mean) / std_safe
        else:
            std_emb = enc_emb
        head_out = run_head_on_std_emb(args.head, std_emb)
        print("Computed head_out (encoder->std->head):", head_out.tolist())

        res = find_best_slice(merged_out, head_out)
        if res is None:
            print("Head output longer than merged output or no match possible.")
        else:
            offset, l2 = res
            print(f"Best-match slice at offset {offset} with L2={l2:.6e}")
            candidate = merged_out.ravel()[offset:offset+head_out.size]
            print("Candidate slice:", candidate.tolist())
            print("Head_out:", head_out.tolist())
        return

    print("Merged model produced an output of size", merged_out.size)
    print("No mean/std or head/encoder provided to interpret it further.")
    if args.verbose:
        print("Merged raw output (first 200 elems):", merged_out.ravel()[:200].tolist())

if __name__ == "__main__":
    main()
# @path: quantize_encoder.py

from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
import glob, os, librosa, numpy as np, sys, onnxruntime as ort

MODEL=sys.argv[1]
OUT=sys.argv[2]
CALIB_DIR=sys.argv[3]

SR=16000
N_FFT=512
HOP=256
N_MELS=96
FRAMES=187

def make_mel(path):
    y,_ = librosa.load(path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).T.astype('float32')

    if mel_db.shape[0] >= FRAMES:
        return mel_db[:FRAMES]

    pad = FRAMES - mel_db.shape[0]
    padv = mel_db.min() if mel_db.shape[0] > 0 else -80.0
    return np.vstack([mel_db, np.full((pad, N_MELS), padv, dtype=np.float32)])

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
INPUT_NAME = sess.get_inputs()[0].name
print("Detected model input:", INPUT_NAME)

class MelReader(CalibrationDataReader):
    def __init__(self, files):
        self.files = files
        self.iter = iter(self.files)

    def get_next(self):
        try:
            f = next(self.iter)
        except StopIteration:
            return None

        mel = make_mel(f)

        return {
            INPUT_NAME: mel[np.newaxis].astype('float32')
        }

    def rewind(self):
        self.iter = iter(self.files)

files = glob.glob(os.path.join(CALIB_DIR, "*.wav"))

dr = MelReader(files)

quantize_static(
    MODEL,
    OUT,
    dr,
    quant_format=QuantFormat.QDQ,
    per_channel=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8
)

print("Wrote", OUT)
# @path: quantize_head.py

import os
import sys
import glob
import numpy as np
import librosa
import onnxruntime as ort

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)

MODEL = sys.argv[1]
OUT = sys.argv[2]
CALIB_DIR = sys.argv[3]

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
INPUT_NAME = sess.get_inputs()[0].name
print("Detected model input:", INPUT_NAME)

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
        mel_db = mel_db[:FRAMES]
    else:
        pad = FRAMES - mel_db.shape[0]
        pad_val = mel_db.min() if mel_db.shape[0] > 0 else -80.0
        mel_db = np.vstack(
            [mel_db, np.full((pad, N_MELS), pad_val, dtype=np.float32)]
        )

    return mel_db

class MelReader(CalibrationDataReader):

    def __init__(self, files):
        self.files = files
        self.index = 0

    def get_next(self):

        if self.index >= len(self.files):
            return None

        f = self.files[self.index]
        self.index += 1

        mel = make_mel(f)

        return {
            INPUT_NAME: mel[np.newaxis, :, :].astype(np.float32)
        }

    def rewind(self):
        self.index = 0

files = sorted(glob.glob(os.path.join(CALIB_DIR, "*.wav")))

if not files:
    raise RuntimeError("No calibration WAV files found in " + CALIB_DIR)

print("Calibration files:", len(files))

reader = MelReader(files)

quantize_static(
    MODEL,
    OUT,
    reader,
    quant_format=QuantFormat.QDQ,
    per_channel=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
)

print("Quantized model written to:", OUT)
# @path: run_inference.py

import argparse
import numpy as np
import librosa
import onnxruntime as ort

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187 

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

    emb_mean = np.load(args.emb_mean)
    emb_std = np.load(args.emb_std)

    emb_std = np.where(emb_std < 1e-8, 1.0, emb_std)

    emb_stdized = (emb - emb_mean) / emb_std

    print("Standardized embedding mean:", emb_stdized.mean())
    print("Standardized embedding std:", emb_stdized.std())

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

    valence, arousal = apply_activation(raw_output, args.activation)

    print("Final Valence:", float(valence))
    print("Final Arousal:", float(arousal))

if __name__ == "__main__":
    main()
# @path: run_two_stage.py

import onnxruntime as ort
import numpy as np
import librosa
import sys

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
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max).T.astype("float32")

    if mel_db.shape[0] >= FRAMES:
        return mel_db[:FRAMES]

    pad = FRAMES - mel_db.shape[0]
    padv = mel_db.min() if mel_db.shape[0] > 0 else -80.0

    return np.vstack([mel_db, np.full((pad, N_MELS), padv, dtype=np.float32)])

if len(sys.argv) != 6:
    print("Usage: run_two_stage.py ENCODER_QDQ HEAD_QDQ MEAN_NPY STD_NPY TEST_WAV")
    sys.exit(1)

ENCODER = sys.argv[1]
HEAD = sys.argv[2]
MEAN = sys.argv[3]
STD = sys.argv[4]
WAV = sys.argv[5]

mean = np.load(MEAN).astype("float32")
std = np.load(STD).astype("float32")

mel = make_mel(WAV)

enc_sess = ort.InferenceSession(ENCODER, providers=["CPUExecutionProvider"])
head_sess = ort.InferenceSession(HEAD, providers=["CPUExecutionProvider"])

enc_input = enc_sess.get_inputs()[0].name
head_input = head_sess.get_inputs()[0].name

emb = enc_sess.run(None, {enc_input: mel[np.newaxis].astype("float32")})[0]
emb = np.array(emb).squeeze().astype("float32")

std_safe = np.where(std == 0.0, 1.0, std)
std_emb = ((emb - mean) / std_safe).astype("float32")

head_out = head_sess.run(None, {head_input: std_emb[np.newaxis]})[0]
head_out = np.array(head_out).squeeze()

print("Head pre-tanh:", head_out.tolist())
print("Head tanh:", np.tanh(head_out).tolist())
# @path: stitch_encoder_head.py

import onnx
from onnx import helper
from onnx.compose import merge_models
import sys

if len(sys.argv) != 5:
    print("Usage: python stitch_encoder_head.py encoder_onnx encoder_name head_onnx out_onnx")
    print("Example: python stitch_encoder_head.py merged_std_correct.onnx merged_std_correct deam_head.onnx merged_va.onnx")
    sys.exit(1)

encoder_path = sys.argv[1]

encoder_name = sys.argv[2]
head_path = sys.argv[3]
out_path = sys.argv[4]

print("Loading encoder:", encoder_path)
enc = onnx.load(encoder_path)
print("Loading head:", head_path)
head = onnx.load(head_path)

enc_out_name = enc.graph.output[0].name
head_in_name = head.graph.input[0].name

print("Connecting encoder output '{}' -> head input '{}'".format(enc_out_name, head_in_name))

merged = merge_models(enc, head, io_map=[(enc_out_name, head_in_name)])

print("Merged model has outputs:", [o.name for o in merged.graph.output])

onnx.save(merged, out_path)
print("Saved merged model:", out_path)
# @path: validation.py
import sys
import numpy as np
import librosa
import onnxruntime as ort

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187

AUDIO_PATH = "test.wav"

ENCODER = "msd_musicnn.onnx"
HEAD = "deam_head.onnx"
MERGED = "merged_std_correct_qdq.onnx"

EMB_MEAN = "emb_mean.npy"
EMB_STD = "emb_std.npy"

def make_mel(path):
    y, _ = librosa.load(path, sr=SR, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db.T.astype(np.float32)

    T = mel_db.shape[0]

    if T >= FRAMES:
        mel_db = mel_db[:FRAMES, :]
    else:
        pad_rows = FRAMES - T
        pad_value = mel_db.min() if T > 0 else -80.0
        mel_db = np.vstack([
            mel_db,
            np.full((pad_rows, N_MELS), pad_value, dtype=np.float32)
        ])

    return mel_db

def python_pipeline(mel):
    enc_sess = ort.InferenceSession(ENCODER, providers=["CPUExecutionProvider"])
    head_sess = ort.InferenceSession(HEAD, providers=["CPUExecutionProvider"])

    enc_input = enc_sess.get_inputs()[0].name
    head_input = head_sess.get_inputs()[0].name

    emb = enc_sess.run(
        None,
        {enc_input: mel[np.newaxis, :, :]}
    )[0]

    emb_mean = np.load(EMB_MEAN)
    emb_std = np.load(EMB_STD)
    emb_std = np.where(emb_std < 1e-8, 1.0, emb_std)

    emb_stdized = (emb - emb_mean) / emb_std

    raw = head_sess.run(
        None,
        {head_input: emb_stdized.astype(np.float32)}
    )[0][0]

    final = np.tanh(raw)

    return final

def merged_pipeline(mel):
    sess = ort.InferenceSession(MERGED, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    out = sess.run(
        None,
        {input_name: mel[np.newaxis, :, :]}
    )[0][0]

    return out

if __name__ == "__main__":
    mel = make_mel(AUDIO_PATH)

    py_out = python_pipeline(mel)
    merged_out = merged_pipeline(mel)

    print("Python reference:", py_out)
    print("Merged model   :", merged_out)

    diff = np.abs(py_out - merged_out)
    print("Absolute diff  :", diff)
    print("Max diff       :", diff.max())
    print("L2 diff        :", float(np.linalg.norm(py_out - merged_out)))

    print("Python out:", py_out.tolist())
    print("Merged out:", merged_out.tolist())

    if merged_out.shape != (2,):
        print(f"ERROR: expected merged output shape (2,), got {merged_out.shape}", file=sys.stderr)
        sys.exit(2)

    if diff.max() >= 1e-3:
        print(f"ERROR: parity check failed, max diff={diff.max():.6e}", file=sys.stderr)
        sys.exit(3)
# @path: verify_merged.py

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
