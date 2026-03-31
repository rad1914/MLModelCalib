# @path: audio_utils.py
from __future__ import annotations

from typing import Sequence

import librosa
import numpy as np

DEFAULT_SR = 16000
DEFAULT_N_FFT = 512
DEFAULT_HOP = 256
DEFAULT_N_MELS = 96
DEFAULT_FRAMES = 187
DEFAULT_POWER = 2.0
DEFAULT_CLIP_MIN = -80.0
DEFAULT_CLIP_MAX = 0.0
STD_FLOOR = 1e-6

def _dim_to_int(value):
    try:
        return int(value)
    except Exception:
        return None

def load_audio(path: str, sr: int = DEFAULT_SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return np.asarray(y, dtype=np.float32)

def make_mel_patch(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
    n_mels: int = DEFAULT_N_MELS,
    frames: int = DEFAULT_FRAMES,
    power: float = DEFAULT_POWER,
    clip_min: float = DEFAULT_CLIP_MIN,
    clip_max: float = DEFAULT_CLIP_MAX,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=power,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, clip_min, clip_max).T.astype(np.float32)

    if mel_db.shape[0] >= frames:
        return mel_db[:frames]

    pad_rows = frames - mel_db.shape[0]
    pad_value = float(mel_db.min()) if mel_db.size else clip_min
    return np.vstack(
        [mel_db, np.full((pad_rows, n_mels), pad_value, dtype=np.float32)]
    )

def prepare_input_for_model(
    patch: np.ndarray,
    model_input_shape: Sequence | None,
    frames: int = DEFAULT_FRAMES,
    n_mels: int = DEFAULT_N_MELS,
) -> np.ndarray:
    arr = np.asarray(patch, dtype=np.float32)[np.newaxis, :, :]

    if not model_input_shape:
        return arr

    shape = list(model_input_shape)

    if len(shape) == 3:
        s1 = _dim_to_int(shape[1])
        s2 = _dim_to_int(shape[2])
        if (s1 is None or s1 == frames) and (s2 is None or s2 == n_mels):
            return arr
        if (s1 is None or s1 == n_mels) and (s2 is None or s2 == frames):
            return arr.transpose(0, 2, 1)

    if len(shape) == 4:
        s1 = _dim_to_int(shape[1])
        s2 = _dim_to_int(shape[2])
        s3 = _dim_to_int(shape[3])
        if (s1 is None or s1 == frames) and (s2 is None or s2 == n_mels):
            return arr[:, :, :, np.newaxis]
        if (s2 is None or s2 == frames) and (s3 is None or s3 == n_mels):
            return arr[:, np.newaxis, :, :]
        if (s1 is None or s1 == n_mels) and (s2 is None or s2 == frames):
            return arr.transpose(0, 2, 1)[:, :, :, np.newaxis]

    return arr

def prepare_vector_for_model(
    vector: np.ndarray,
    model_input_shape: Sequence | None,
) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)

    if not model_input_shape:
        return vec[np.newaxis, :]

    shape = list(model_input_shape)

    if len(shape) == 1:
        return vec

    if len(shape) == 2:
        s0 = _dim_to_int(shape[0])
        s1 = _dim_to_int(shape[1])
        if s1 is None or s1 == vec.size:
            return vec[np.newaxis, :]
        if s0 is None or s0 == vec.size:
            return vec[:, np.newaxis]

    return vec[np.newaxis, :]

def standardize_embedding(
    emb: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    floor: float = STD_FLOOR,
) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    std_safe = np.maximum(std, floor)
    return ((emb - mean) / std_safe).astype(np.float32)
# @path: compute_calib_stats.py

from __future__ import annotations
import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import onnxruntime as ort

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    STD_FLOOR,
    load_audio,
    make_mel_patch,
    prepare_input_for_model,
)

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
    p.add_argument('--power', type=float, default=DEFAULT_POWER, help='Power for mel spectrogram (2.0 for power)')
    p.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    return p.parse_args()

def list_wavs(calib_dir: str) -> List[str]:
    files = sorted([f for f in os.listdir(calib_dir) if f.lower().endswith('.wav')])
    return files

def prepare_input_for_model(patch: np.ndarray, model_input_shape, frames: int, n_mels: int) -> np.ndarray:
    return prepare_input_for_model(patch, model_input_shape, frames=frames, n_mels=n_mels)

def open_session(model_path: str) -> ort.InferenceSession:
    try:
        return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print("Error: Failed to open ONNX model:", e, file=sys.stderr)
        raise

def compute_stats_from_embeddings(emb_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
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
    for i, fname in enumerate(wavs, 1):
        path = os.path.join(args.calib, fname)
        try:
            y = load_audio(path, sr=args.sr)
        except Exception as e:
            print(f"Warning: failed to load {fname}: {e}", file=sys.stderr)
            continue

        patch = make_mel_patch(
            y,
            sr=args.sr,
            n_fft=args.n_fft,
            hop=args.hop,
            n_mels=args.n_mels,
            frames=args.frames,
            power=args.power,
        )
        inp = prepare_input_for_model(patch, in_shape, frames=args.frames, n_mels=args.n_mels).astype(np.float32)
        try:
            out = sess.run(None, {in_name: inp})
        except Exception as e:
            print(f"Warning: model inference failed on {fname}: {e}", file=sys.stderr)
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
            print(f"[{i}/{len(wavs)}] {fname}: patches=1 total_patches={total_patches}")

    if not emb_list:
        print("ERROR: No embeddings were produced. Check model input shape and calibration audio.", file=sys.stderr)
        sys.exit(4)

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
    if len(head.graph.output) != 1:
        raise RuntimeError(f"Head must have exactly 1 output, got {len(head.graph.output)}")

    enc_out = enc.graph.output[0].name
    head_in  = head.graph.input[0].name

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

    head_internal = set()
    for node in head.graph.node:
        head_internal.update(node.input)
        head_internal.update(node.output)
    if head_in not in head_internal:
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

    sub_nodes = [n for n in merged.graph.node if n.op_type == "Sub" and len(n.output) == 1 and n.output[0] == sub_out]
    div_nodes = [n for n in merged.graph.node if n.op_type == "Div" and len(n.output) == 1 and n.output[0] == div_out]
    if not sub_nodes or not div_nodes:
        raise RuntimeError("Failed to insert standardization nodes")
    if enc_out not in sub_nodes[0].input:
        raise RuntimeError("Encoder output is not feeding the Sub node")
    if sub_out not in div_nodes[0].input:
        raise RuntimeError("Sub output is not feeding the Div node")
    head_consumers = [n for n in merged.graph.node if n.name.startswith(prefix) and div_out in n.input]
    if not head_consumers:
        raise RuntimeError("Standardized embedding does not reach the head")

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

    if len(merged.graph.output) != len(head.graph.output):
        raise RuntimeError("Failed to propagate head outputs into merged graph")

    def _dims(vi):
        return [
            d.dim_value if d.HasField("dim_value") else None
            for d in vi.type.tensor_type.shape.dim
        ]

    out_dims = _dims(merged.graph.output[0])
    known_out_dims = [d for d in out_dims if d is not None]
    if known_out_dims and 2 not in known_out_dims:
        raise RuntimeError(f"Head output does not look like a 2-value VA tensor: {out_dims}")

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

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
import glob
import os
import sys

import numpy as np
import onnxruntime as ort

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    load_audio,
    make_mel_patch,
    prepare_input_for_model,
)

MODEL = sys.argv[1]
OUT = sys.argv[2]
CALIB_DIR = sys.argv[3]

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
INPUT_NAME = sess.get_inputs()[0].name
INPUT_SHAPE = sess.get_inputs()[0].shape
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

        y = load_audio(f, sr=DEFAULT_SR)
        mel = make_mel_patch(
            y,
            sr=DEFAULT_SR,
            n_fft=DEFAULT_N_FFT,
            hop=DEFAULT_HOP,
            n_mels=DEFAULT_N_MELS,
            frames=DEFAULT_FRAMES,
            power=DEFAULT_POWER,
        )
        inp = prepare_input_for_model(
            mel,
            INPUT_SHAPE,
            frames=DEFAULT_FRAMES,
            n_mels=DEFAULT_N_MELS,
        )

        return {
            INPUT_NAME: inp.astype(np.float32)
        }

    def rewind(self):
        self.iter = iter(self.files)

files = sorted(glob.glob(os.path.join(CALIB_DIR, "*.wav")))

if not files:
    raise RuntimeError("No calibration WAV files found in " + CALIB_DIR)

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

import argparse
import glob
import os

import numpy as np
import onnxruntime as ort

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    load_audio,
    make_mel_patch,
    prepare_input_for_model,
    prepare_vector_for_model,
    standardize_embedding,
)

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("out")
parser.add_argument("calib_dir")
parser.add_argument("--encoder", required=True, help="Encoder ONNX used to produce calibration embeddings")
parser.add_argument("--emb-mean", required=True, help="Calibration embedding mean .npy")
parser.add_argument("--emb-std", required=True, help="Calibration embedding std .npy")
args = parser.parse_args()

MODEL = args.model
OUT = args.out
CALIB_DIR = args.calib_dir
ENCODER_PATH = args.encoder
EMB_MEAN_PATH = args.emb_mean
EMB_STD_PATH = args.emb_std

enc_sess = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
enc_input_name = enc_sess.get_inputs()[0].name
enc_input_shape = enc_sess.get_inputs()[0].shape
head_sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
HEAD_INPUT_NAME = head_sess.get_inputs()[0].name
HEAD_INPUT_SHAPE = head_sess.get_inputs()[0].shape
print("Detected head input:", HEAD_INPUT_NAME)

emb_mean = np.load(EMB_MEAN_PATH).astype(np.float32)
emb_std = np.load(EMB_STD_PATH).astype(np.float32)

class MelReader(CalibrationDataReader):

    def __init__(self, files):
        self.files = files
        self.index = 0

    def get_next(self):

        if self.index >= len(self.files):
            return None

        f = self.files[self.index]
        self.index += 1

        y = load_audio(f, sr=DEFAULT_SR)
        mel = make_mel_patch(
            y,
            sr=DEFAULT_SR,
            n_fft=DEFAULT_N_FFT,
            hop=DEFAULT_HOP,
            n_mels=DEFAULT_N_MELS,
            frames=DEFAULT_FRAMES,
            power=DEFAULT_POWER,
        )
        enc_inp = prepare_input_for_model(
            mel,
            enc_input_shape,
            frames=DEFAULT_FRAMES,
            n_mels=DEFAULT_N_MELS,
        ).astype(np.float32)
        emb = np.asarray(
            enc_sess.run(None, {enc_input_name: enc_inp})[0]
        ).squeeze()
        std_emb = standardize_embedding(emb, emb_mean, emb_std)
        head_inp = prepare_vector_for_model(std_emb, HEAD_INPUT_SHAPE)

        return {
            HEAD_INPUT_NAME: head_inp.astype(np.float32)
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
# @path: quantize_merged.py

import argparse
import glob
import os

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    load_audio,
    make_mel_patch,
    prepare_input_for_model,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("calib_dir")
    args = parser.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    print("Detected merged input:", input_name)

    files = sorted(glob.glob(os.path.join(args.calib_dir, "*.wav")))
    if not files:
        raise RuntimeError("No calibration WAV files found in " + args.calib_dir)

    class MelReader(CalibrationDataReader):
        def __init__(self, paths):
            self.paths = paths
            self.index = 0

        def get_next(self):
            if self.index >= len(self.paths):
                return None
            f = self.paths[self.index]
            self.index += 1
            y = load_audio(f, sr=DEFAULT_SR)
            mel = make_mel_patch(
                y,
                sr=DEFAULT_SR,
                n_fft=DEFAULT_N_FFT,
                hop=DEFAULT_HOP,
                n_mels=DEFAULT_N_MELS,
                frames=DEFAULT_FRAMES,
                power=DEFAULT_POWER,
            )
            inp = prepare_input_for_model(
                mel,
                input_shape,
                frames=DEFAULT_FRAMES,
                n_mels=DEFAULT_N_MELS,
            )
            return {input_name: inp.astype(np.float32)}

        def rewind(self):
            self.index = 0

    reader = MelReader(files)

    quantize_static(
        args.model,
        args.out,
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )

    print("Quantized merged model written to:", args.out)

if __name__ == "__main__":
    main()
# @path: run_inference.py

import argparse
import numpy as np
import onnxruntime as ort

from audio_utils import load_audio, make_mel_patch, prepare_input_for_model

def apply_activation(output, mode):
    output = np.asarray(output, dtype=np.float32).reshape(-1)
    if output.size != 2:
        raise RuntimeError(f"Expected 2 output values, got shape {output.shape}")
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--activation", default="raw",
                        choices=["raw", "tanh", "sigmoid"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Loading audio:", args.audio)
    y = load_audio(args.audio)

    mel_patch = make_mel_patch(y)
    print("Mel patch shape:", mel_patch.shape)

    sess = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"]
    )

    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    model_input = prepare_input_for_model(
        mel_patch,
        input_meta.shape,
    )

    out = sess.run(
        None,
        {input_name: model_input.astype(np.float32)}
    )[0]
    raw_output = np.asarray(out).squeeze().astype(np.float32)
    if raw_output.size != 2:
        raise RuntimeError(f"Expected merged model to return 2 values, got shape {raw_output.shape}")

    print("Raw model output:", raw_output.tolist())

    valence, arousal = apply_activation(raw_output, args.activation)

    print("Final Valence:", float(valence))
    print("Final Arousal:", float(arousal))

if __name__ == "__main__":
    main()
# @path: run_two_stage.py

import onnxruntime as ort
import numpy as np
import sys

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    load_audio,
    make_mel_patch,
    prepare_input_for_model,
    prepare_vector_for_model,
    standardize_embedding,
)

def make_mel(path):
    y = load_audio(path, sr=DEFAULT_SR)
    return make_mel_patch(
        y,
        sr=DEFAULT_SR,
        n_fft=DEFAULT_N_FFT,
        hop=DEFAULT_HOP,
        n_mels=DEFAULT_N_MELS,
        frames=DEFAULT_FRAMES,
        power=DEFAULT_POWER,
    )

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
enc_input_shape = enc_sess.get_inputs()[0].shape
head_input_shape = head_sess.get_inputs()[0].shape

enc_inp = prepare_input_for_model(
    mel,
    enc_input_shape,
    frames=DEFAULT_FRAMES,
    n_mels=DEFAULT_N_MELS,
).astype(np.float32)

emb = enc_sess.run(None, {enc_input: enc_inp})[0]
emb = np.array(emb).squeeze().astype("float32")

std_emb = standardize_embedding(emb, mean, std)
head_inp = prepare_vector_for_model(std_emb, head_input_shape)

head_out = head_sess.run(None, {head_input: head_inp.astype(np.float32)})[0]
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
import onnxruntime as ort

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    load_audio,
    make_mel_patch,
    prepare_input_for_model,
    prepare_vector_for_model,
    standardize_embedding,
)

AUDIO_PATH = "test.wav"

ENCODER = "msd_musicnn.onnx"
HEAD = "deam_head.onnx"
MERGED = "merged_std_correct_qdq.onnx"

EMB_MEAN = "emb_mean.npy"
EMB_STD = "emb_std.npy"

def make_mel(path):
    y = load_audio(path, sr=DEFAULT_SR)
    return make_mel_patch(
        y,
        sr=DEFAULT_SR,
        n_fft=DEFAULT_N_FFT,
        hop=DEFAULT_HOP,
        n_mels=DEFAULT_N_MELS,
        frames=DEFAULT_FRAMES,
        power=DEFAULT_POWER,
    )

def python_pipeline(mel):
    enc_sess = ort.InferenceSession(ENCODER, providers=["CPUExecutionProvider"])
    head_sess = ort.InferenceSession(HEAD, providers=["CPUExecutionProvider"])

    enc_input = enc_sess.get_inputs()[0].name
    head_input = head_sess.get_inputs()[0].name
    enc_shape = enc_sess.get_inputs()[0].shape
    head_shape = head_sess.get_inputs()[0].shape

    enc_inp = prepare_input_for_model(
        mel,
        enc_shape,
        frames=DEFAULT_FRAMES,
        n_mels=DEFAULT_N_MELS,
    )

    emb = enc_sess.run(
        None,
        {enc_input: enc_inp.astype(np.float32)}
    )[0]

    emb_mean = np.load(EMB_MEAN)
    emb_std = np.load(EMB_STD)
    emb_std = np.asarray(emb_std, dtype=np.float32)

    emb_stdized = standardize_embedding(emb, emb_mean, emb_std)
    head_inp = prepare_vector_for_model(emb_stdized, head_shape)

    raw = head_sess.run(
        None,
        {head_input: head_inp.astype(np.float32)}
    )[0][0]

    return np.asarray(raw, dtype=np.float32).reshape(-1)

def merged_pipeline(mel):
    sess = ort.InferenceSession(MERGED, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    inp = prepare_input_for_model(
        mel,
        input_shape,
        frames=DEFAULT_FRAMES,
        n_mels=DEFAULT_N_MELS,
    )

    out = sess.run(
        None,
        {input_name: inp.astype(np.float32)}
    )[0][0]

    return np.asarray(out, dtype=np.float32).reshape(-1)

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

    if diff.max() >= 1e-4:
        print(f"ERROR: parity check failed, max diff={diff.max():.6e}", file=sys.stderr)
        sys.exit(3)
# @path: verify_merged.py

import argparse
import numpy as np
import onnxruntime as ort
from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    load_audio,
    make_mel_patch,
    prepare_input_for_model,
)

def make_mel(path):
    y = load_audio(path, sr=DEFAULT_SR)
    return make_mel_patch(
        y,
        sr=DEFAULT_SR,
        n_fft=DEFAULT_N_FFT,
        hop=DEFAULT_HOP,
        n_mels=DEFAULT_N_MELS,
        frames=DEFAULT_FRAMES,
        power=DEFAULT_POWER,
    )

def run_onnx(model_path, mel):
    sess = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    inp = prepare_input_for_model(
        mel,
        input_shape,
        frames=DEFAULT_FRAMES,
        n_mels=DEFAULT_N_MELS,
    )

    outputs = sess.run(
        None,
        {input_name: inp.astype(np.float32)}
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
