# @path: audio_utils.py
from __future__ import annotations

from functools import lru_cache
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

def _dim_matches(dim, expected: int) -> bool:
    return dim is None or dim == expected

def _dim_to_int(value):
    try:
        return int(value)
    except Exception:
        return None

def _has_known_positive_dim(dim) -> bool:
    return dim is not None and int(dim) > 0

def _shape_has_any_known_positive_dim(shape: Sequence | None) -> bool:
    return any(_has_known_positive_dim(_dim_to_int(dim)) for dim in (shape or []))

def _normalize_shape(shape: Sequence | None):
    if not shape:
        return None
    return list(shape)

@lru_cache(maxsize=None)
def get_cpu_session(model_path: str):
    import onnxruntime as ort

    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

def sanitize_audio(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Audio buffer is empty")
    if not np.isfinite(arr).all():
        raise ValueError("Audio buffer contains NaN or infinite values")
    return arr

def load_audio(path: str, sr: int = DEFAULT_SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return sanitize_audio(y)

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
    y = sanitize_audio(y)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=power,
    )
    if mel.size == 0:
        raise ValueError("Mel spectrogram is empty")

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, clip_min, clip_max).T.astype(np.float32)

    if mel_db.shape[0] >= frames:
        return mel_db[:frames]

    pad_rows = frames - mel_db.shape[0]
    pad_value = float(mel_db.min()) if mel_db.size else clip_min
    return np.vstack(
        [mel_db, np.full((pad_rows, n_mels), pad_value, dtype=np.float32)]
    )

def load_mel_patch(
    path: str,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
    n_mels: int = DEFAULT_N_MELS,
    frames: int = DEFAULT_FRAMES,
    power: float = DEFAULT_POWER,
    clip_min: float = DEFAULT_CLIP_MIN,
    clip_max: float = DEFAULT_CLIP_MAX,
) -> np.ndarray:
    y = load_audio(path, sr=sr)
    return make_mel_patch(
        y,
        sr=sr,
        n_fft=n_fft,
        hop=hop,
        n_mels=n_mels,
        frames=frames,
        power=power,
        clip_min=clip_min,
        clip_max=clip_max,
    )

def prepare_input_for_model(
    patch: np.ndarray,
    model_input_shape: Sequence | None,
    frames: int = DEFAULT_FRAMES,
    n_mels: int = DEFAULT_N_MELS,
) -> np.ndarray:
    patch = np.asarray(patch, dtype=np.float32)
    if patch.ndim != 2:
        raise ValueError(f"Expected 2D mel patch, got shape {patch.shape}")

    arr = patch[np.newaxis, :, :]

    shape = _normalize_shape(model_input_shape)
    if not shape:
        return arr

    if len(shape) == 3:
        s1 = _dim_to_int(shape[1])
        s2 = _dim_to_int(shape[2])
        candidates = []
        if _dim_matches(s1, frames) and _dim_matches(s2, n_mels):
            candidates.append(arr)
        if _dim_matches(s1, n_mels) and _dim_matches(s2, frames):
            candidates.append(arr.transpose(0, 2, 1))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous 3D model input shape: {model_input_shape}")
        if _shape_has_any_known_positive_dim(shape[1:]):
            raise ValueError(
                f"Model input shape {model_input_shape} is incompatible with mel patch "
                f"({frames}, {n_mels})"
            )
        return arr

    if len(shape) == 4:
        s1 = _dim_to_int(shape[1])
        s2 = _dim_to_int(shape[2])
        s3 = _dim_to_int(shape[3])
        candidates = []
        if _dim_matches(s1, frames) and _dim_matches(s2, n_mels):
            candidates.append(arr[:, :, :, np.newaxis])
        if _dim_matches(s2, frames) and _dim_matches(s3, n_mels):
            candidates.append(arr[:, np.newaxis, :, :])
        if _dim_matches(s1, n_mels) and _dim_matches(s2, frames):
            candidates.append(arr.transpose(0, 2, 1)[:, :, :, np.newaxis])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous 4D model input shape: {model_input_shape}")
        if _shape_has_any_known_positive_dim(shape[1:]):
            raise ValueError(
                f"Model input shape {model_input_shape} is incompatible with mel patch "
                f"({frames}, {n_mels})"
            )
        return arr

    return arr

def build_model_input_from_path(
    path: str,
    model_input_shape: Sequence | None,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
    n_mels: int = DEFAULT_N_MELS,
    frames: int = DEFAULT_FRAMES,
    power: float = DEFAULT_POWER,
    clip_min: float = DEFAULT_CLIP_MIN,
    clip_max: float = DEFAULT_CLIP_MAX,
) -> np.ndarray:
    mel = load_mel_patch(
        path,
        sr=sr,
        n_fft=n_fft,
        hop=hop,
        n_mels=n_mels,
        frames=frames,
        power=power,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    return prepare_input_for_model(
        mel,
        model_input_shape,
        frames=frames,
        n_mels=n_mels,
    )

def prepare_vector_for_model(
    vector: np.ndarray,
    model_input_shape: Sequence | None,
) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)

    shape = _normalize_shape(model_input_shape)
    if not shape:
        return vec[np.newaxis, :]

    if len(shape) == 1:
        s0 = _dim_to_int(shape[0])
        if _has_known_positive_dim(s0) and s0 != vec.size:
            raise ValueError(
                f"Vector length {vec.size} does not match model input shape {model_input_shape}"
            )
        return vec

    if len(shape) == 2:
        s0 = _dim_to_int(shape[0])
        s1 = _dim_to_int(shape[1])
        candidates = []
        if s1 is None or s1 == vec.size:
            candidates.append(vec[np.newaxis, :])
        if s0 is None or s0 == vec.size:
            candidates.append(vec[:, np.newaxis])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous 2D model input shape: {model_input_shape}")
        if _shape_has_any_known_positive_dim(shape):
            raise ValueError(
                f"Vector length {vec.size} is incompatible with model input shape {model_input_shape}"
            )
        return vec[np.newaxis, :]

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

    if mean.size != std.size:
        raise ValueError(f"Mean/std shape mismatch: {mean.shape} vs {std.shape}")
    if emb.size != mean.size:
        raise ValueError(
            f"Embedding length {emb.size} does not match calibration stats length {mean.size}"
        )

    std_safe = np.maximum(std, floor)
    return ((emb - mean) / std_safe).astype(np.float32)
# @path: compute_calib_stats.py
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    STD_FLOOR,
    build_model_input_from_path,
    get_cpu_session,
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
    p.add_argument('--min-samples', type=int, default=5, help='Minimum successfully embedded clips required')
    p.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    return p.parse_args()

def list_wavs(calib_dir: str) -> List[str]:
    return sorted(
        os.path.join(calib_dir, f)
        for f in os.listdir(calib_dir)
        if f.lower().endswith('.wav')
    )

def open_session(model_path: str):
    try:
        return get_cpu_session(model_path)
    except Exception as e:
        print("Error: Failed to open ONNX model:", e, file=sys.stderr)
        raise

def compute_stats_from_embeddings(emb_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
    embs = np.stack(emb_list, axis=0)
    mean = embs.mean(axis=0)
    std = embs.std(axis=0)
    std = np.maximum(std, STD_FLOOR)
    return mean.astype(np.float32), std.astype(np.float32), int(embs.shape[0])

def _output_dim_hint(output_shape) -> int | None:
    dims = []
    for dim in output_shape or []:
        try:
            dims.append(int(dim))
        except Exception:
            dims.append(None)

    for dim in reversed(dims):
        if dim is not None and dim > 0:
            return dim
    return None

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
    out_meta = sess.get_outputs()[0]
    expected_dim = _output_dim_hint(out_meta.shape)

    if args.verbose:
        print("Model input name:", in_name, "shape:", in_shape)
        print("Model output name:", out_meta.name, "shape:", out_meta.shape)
        if expected_dim is not None:
            print("Expected embedding dim from model output metadata:", expected_dim)

    emb_list = []
    failed = 0
    total_patches = 0
    emb_dim = expected_dim

    for i, path in enumerate(wavs, 1):
        fname = os.path.basename(path)
        try:
            inp = build_model_input_from_path(
                path,
                in_shape,
                sr=args.sr,
                n_fft=args.n_fft,
                hop=args.hop,
                n_mels=args.n_mels,
                frames=args.frames,
                power=args.power,
            ).astype(np.float32)
            out = sess.run(None, {in_name: inp})
        except Exception as e:
            failed += 1
            print(f"Warning: failed on {fname}: {e}", file=sys.stderr)
            continue

        if not out:
            failed += 1
            print(f"Warning: unexpected encoder output for {fname}", file=sys.stderr)
            continue

        emb = np.asarray(out[0], dtype=np.float32).reshape(-1)
        if emb_dim is None:
            emb_dim = int(emb.size)
            if args.verbose:
                print("Inferred embedding dim from first successful sample:", emb_dim)
        elif emb.size != emb_dim:
            raise RuntimeError(
                f"Embedding dim mismatch for {fname}: got {emb.size}, expected {emb_dim}"
            )

        emb_list.append(emb)
        total_patches += 1

        if args.verbose:
            print(f"[{i}/{len(wavs)}] {fname}: patches=1 total_patches={total_patches}")

    if not emb_list:
        print("ERROR: No embeddings were produced. Check model input shape and calibration audio.", file=sys.stderr)
        sys.exit(4)

    if len(emb_list) < max(1, args.min_samples):
        print(
            f"ERROR: Only {len(emb_list)} calibration samples succeeded; "
            f"minimum required is {args.min_samples}.",
            file=sys.stderr,
        )
        sys.exit(5)

    mean, std, n_samples = compute_stats_from_embeddings(emb_list)

    np.save(args.out_mean, mean)
    np.save(args.out_std, std)
    print(f"Saved {args.out_mean} and {args.out_std}")
    print(f"Samples (patches): {n_samples}, embedding dim: {mean.shape[0]}")
    print(f"Failed files: {failed}")
    print(f"Embedding mean(mean) = {float(mean.mean()):.6f}, embedding mean(std) = {float(std.mean()):.6f}")

if __name__ == '__main__':
    main()
# @path: merged_runner.py
import argparse
from typing import Optional

import numpy as np
import sys

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    get_cpu_session,
    load_mel_patch,
    prepare_vector_for_model,
    standardize_embedding,
)

def make_mel(path):
    return load_mel_patch(
        path,
        sr=DEFAULT_SR,
        n_fft=DEFAULT_N_FFT,
        hop=DEFAULT_HOP,
        n_mels=DEFAULT_N_MELS,
        frames=DEFAULT_FRAMES,
        power=DEFAULT_POWER,
    )

def run_onnx(model_path: str, input_arr: np.ndarray) -> np.ndarray:
    sess = get_cpu_session(model_path)
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: input_arr.astype(np.float32)})
    return np.asarray(out[0], dtype=np.float32).squeeze()

def run_encoder_get_emb(encoder_path: str, mel: np.ndarray) -> np.ndarray:
    return run_onnx(encoder_path, mel[np.newaxis]).squeeze()

def run_head_on_std_emb(head_path: str, std_emb: np.ndarray) -> np.ndarray:
    return run_onnx(head_path, std_emb[np.newaxis]).squeeze()

def find_best_slice(big: np.ndarray, small: np.ndarray):
    big1 = np.asarray(big, dtype=np.float32).ravel()
    small1 = np.asarray(small, dtype=np.float32).ravel()
    n, m = big1.size, small1.size
    if m > n:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(big1, m)
    dists = np.linalg.norm(windows - small1, axis=1)
    best_i = int(np.argmin(dists))
    return best_i, float(dists[best_i])

def safe_load_np(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in {path}")
    return arr

def _output_dim_hint(output_shape) -> int | None:
    dims = []
    for dim in output_shape or []:
        try:
            dims.append(int(dim))
        except Exception:
            dims.append(None)

    for dim in reversed(dims):
        if dim is not None and dim > 0:
            return dim
    return None

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

    merged_sess = get_cpu_session(args.merged)
    merged_input = merged_sess.get_inputs()[0]
    merged_hint = _output_dim_hint(merged_sess.get_outputs()[0].shape)

    print("Running merged model:", args.merged)
    merged_out = np.asarray(
        merged_sess.run(None, {merged_input.name: mel[np.newaxis].astype(np.float32)})[0],
        dtype=np.float32,
    ).squeeze()
    print("merged_out shape:", merged_out.shape, "size:", merged_out.size)

    if args.verbose:
        print("merged_out dtype:", merged_out.dtype)
        print("merged_out first_10:", merged_out.ravel()[:10].tolist())

    mean = safe_load_np(args.mean) if args.mean else None
    std = safe_load_np(args.std) if args.std else None

    if mean is not None:
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    if std is not None:
        std = np.asarray(std, dtype=np.float32).reshape(-1)

    if mean is not None and merged_out.size == mean.size:
        if args.verbose:
            print("Merged output length equals mean length -> interpreting as embedding")
            print("mean.shape:", mean.shape, "std.shape:", None if std is None else std.shape)
        if std is None:
            print("ERROR: std missing for embedding interpretation", file=sys.stderr)
            sys.exit(2)
        if std.size != mean.size:
            print("ERROR: std size mismatch", file=sys.stderr)
            sys.exit(3)

        emb = merged_out.astype(np.float32).reshape(mean.shape)
        std_emb = standardize_embedding(emb, mean, std)
        if args.head is None:
            print("HEAD model is required to process standardized embedding. Provide --head path.")
            sys.exit(2)
        head_out = run_head_on_std_emb(args.head, std_emb)
        print("Valence/Arousal from head (merged->std_emb->head):", head_out.tolist())
        return

    if merged_hint == 2 and merged_out.size == 2:
        va = merged_out.ravel()
        print("Merged model produced valence/arousal:", va.tolist())
        return

    if args.head and args.encoder:
        print("Merged output is not a 2-value VA tensor and does not match the embedding stats size. Running encoder->head parity search.")
        enc_emb = np.asarray(run_encoder_get_emb(args.encoder, mel), dtype=np.float32).squeeze()
        if mean is not None:
            if std is None:
                print("ERROR: std missing for embedding standardization", file=sys.stderr)
                sys.exit(2)
            if mean.size != enc_emb.size or std.size != enc_emb.size:
                print("ERROR: calibration vectors do not match encoder embedding size", file=sys.stderr)
                sys.exit(3)
            std_emb = standardize_embedding(enc_emb, mean, std)
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
            candidate = merged_out.ravel()[offset:offset + head_out.size]
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

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    build_model_input_from_path,
    get_cpu_session,
)

MODEL = sys.argv[1]
OUT = sys.argv[2]
CALIB_DIR = sys.argv[3]

sess = get_cpu_session(MODEL)
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

        inp = build_model_input_from_path(
            f,
            INPUT_SHAPE,
            sr=DEFAULT_SR,
            n_fft=DEFAULT_N_FFT,
            hop=DEFAULT_HOP,
            n_mels=DEFAULT_N_MELS,
            frames=DEFAULT_FRAMES,
            power=DEFAULT_POWER,
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
    build_model_input_from_path,
    get_cpu_session,
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

enc_sess = get_cpu_session(ENCODER_PATH)
enc_input_name = enc_sess.get_inputs()[0].name
enc_input_shape = enc_sess.get_inputs()[0].shape
head_sess = get_cpu_session(MODEL)
HEAD_INPUT_NAME = head_sess.get_inputs()[0].name
HEAD_INPUT_SHAPE = head_sess.get_inputs()[0].shape
print("Detected head input:", HEAD_INPUT_NAME)

def _load_stats(path):
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise RuntimeError(f"Empty calibration stats in {path}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"Non-finite calibration stats in {path}")
    return arr

emb_mean = _load_stats(EMB_MEAN_PATH)
emb_std = _load_stats(EMB_STD_PATH)

class MelReader(CalibrationDataReader):

    def __init__(self, files):
        self.files = files
        self.index = 0

    def get_next(self):

        if self.index >= len(self.files):
            return None

        f = self.files[self.index]
        self.index += 1

        enc_inp = build_model_input_from_path(
            f,
            enc_input_shape,
            sr=DEFAULT_SR,
            n_fft=DEFAULT_N_FFT,
            hop=DEFAULT_HOP,
            n_mels=DEFAULT_N_MELS,
            frames=DEFAULT_FRAMES,
            power=DEFAULT_POWER,
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
    build_model_input_from_path,
    get_cpu_session,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("calib_dir")
    args = parser.parse_args()

    sess = get_cpu_session(args.model)
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
            inp = build_model_input_from_path(
                f,
                input_shape,
                sr=DEFAULT_SR,
                n_fft=DEFAULT_N_FFT,
                hop=DEFAULT_HOP,
                n_mels=DEFAULT_N_MELS,
                frames=DEFAULT_FRAMES,
                power=DEFAULT_POWER,
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
# @path: run_two_stage.py
import sys

import numpy as np

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    get_cpu_session,
    load_mel_patch,
    prepare_input_for_model,
    prepare_vector_for_model,
    standardize_embedding,
)

def make_mel(path):
    return load_mel_patch(
        path,
        sr=DEFAULT_SR,
        n_fft=DEFAULT_N_FFT,
        hop=DEFAULT_HOP,
        n_mels=DEFAULT_N_MELS,
        frames=DEFAULT_FRAMES,
        power=DEFAULT_POWER,
    )

def _load_stats(path):
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite calibration stats found in {path}")
    return arr

if len(sys.argv) != 6:
    print("Usage: run_two_stage.py ENCODER_QDQ HEAD_QDQ MEAN_NPY STD_NPY TEST_WAV")
    sys.exit(1)

ENCODER = sys.argv[1]
HEAD = sys.argv[2]
MEAN = sys.argv[3]
STD = sys.argv[4]
WAV = sys.argv[5]

mean = _load_stats(MEAN)
std = _load_stats(STD)

mel = make_mel(WAV)

enc_sess = get_cpu_session(ENCODER)
head_sess = get_cpu_session(HEAD)

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
# @path: validation.py
import sys
import numpy as np

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    get_cpu_session,
    load_mel_patch,
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
    return load_mel_patch(
        path,
        sr=DEFAULT_SR,
        n_fft=DEFAULT_N_FFT,
        hop=DEFAULT_HOP,
        n_mels=DEFAULT_N_MELS,
        frames=DEFAULT_FRAMES,
        power=DEFAULT_POWER,
    )

def _load_stats(path):
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite calibration stats found in {path}")
    return arr

def python_pipeline(mel):
    enc_sess = get_cpu_session(ENCODER)
    head_sess = get_cpu_session(HEAD)

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

    emb_mean = _load_stats(EMB_MEAN)
    emb_std = _load_stats(EMB_STD)

    emb_standardized = standardize_embedding(emb, emb_mean, emb_std)
    head_inp = prepare_vector_for_model(emb_standardized, head_shape)

    raw = np.asarray(
        head_sess.run(
            None,
            {head_input: head_inp.astype(np.float32)}
        )[0],
        dtype=np.float32,
    ).squeeze()

    return raw.reshape(-1)

def merged_pipeline(mel):
    sess = get_cpu_session(MERGED)
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    inp = prepare_input_for_model(
        mel,
        input_shape,
        frames=DEFAULT_FRAMES,
        n_mels=DEFAULT_N_MELS,
    )

    out = np.asarray(
        sess.run(
            None,
            {input_name: inp.astype(np.float32)}
        )[0],
        dtype=np.float32,
    ).squeeze()

    return out.reshape(-1)

if __name__ == "__main__":
    mel = make_mel(AUDIO_PATH)

    py_out = python_pipeline(mel)
    merged_out = merged_pipeline(mel)

    print("Python reference:", py_out)
    print("Merged model    :", merged_out)

    diff = np.abs(py_out - merged_out)
    print("Absolute diff  :", diff)
    print("Max diff        :", diff.max())
    print("L2 diff         : ", float(np.linalg.norm(py_out - merged_out)))

    print("Python out:", py_out.tolist())
    print("Merged out:", merged_out.tolist())

    if merged_out.shape != (2,):
        print(f"ERROR: expected merged output shape (2,), got {merged_out.shape}", file=sys.stderr)
        sys.exit(2)

    if diff.max() >= 1e-4:
        print(f"ERROR: parity check failed, max diff={diff.max():.6e}", file=sys.stderr)
        sys.exit(3)
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
# @path: merge_onnx_with_standardization.py
import sys
import numpy as np
import onnx
from onnx import ModelProto, helper, numpy_helper

from audio_utils import STD_FLOOR

# Patched: Added allow_pickle=False and finite checks to prevent the model from choking on trash data.

def _shape_dims(value_info):
    return [
        d.dim_value if d.HasField("dim_value") else None
        for d in value_info.type.tensor_type.shape.dim
    ]

def _known_positive_dims(shape):
    dims = []
    for dim in shape or []:
        try:
            dim = int(dim)
        except Exception:
            dim = None
        if dim is not None and dim > 0:
            dims.append(dim)
    return dims

def _validate_encoder_output_shape(enc_out_shape, emb_dim):
    if not enc_out_shape:
        return

    dims = []
    for dim in enc_out_shape:
        try:
            dims.append(int(dim))
        except Exception:
            dims.append(None)

    known = [dim for dim in dims if dim is not None and dim > 0]
    if not known:
        return

    # Accept common shapes such as [emb_dim], [1, emb_dim], [batch, emb_dim].
    if known[-1] == emb_dim:
        return

    raise RuntimeError(
        f"Encoder output shape {enc_out_shape} is incompatible with embedding dim {emb_dim}"
    )

def main():
    if len(sys.argv) != 6:
        print("Usage: python merge_onnx_with_standardization.py encoder.onnx head.onnx emb_mean.npy emb_std.npy out_merged.onnx")
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

    enc_out_vi = enc.graph.output[0]
    head_in_vi = head.graph.input[0]
    enc_out = enc_out_vi.name
    head_in = head_in_vi.name

    # Patch applied: explicit allow_pickle=False
    mean = np.load(mean_path, allow_pickle=False).astype(np.float32).reshape(-1)
    std = np.load(std_path, allow_pickle=False).astype(np.float32).reshape(-1)

    if mean.shape != std.shape:
        raise RuntimeError(f"Mean/std shape mismatch: {mean.shape} vs {std.shape}")
    if mean.ndim != 1:
        raise RuntimeError(f"Expected 1D embedding stats, got mean.ndim={mean.ndim}")
    if mean.size == 0:
        raise RuntimeError("Empty embedding statistics are not allowed")

    # Patch applied: finite value validation
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("Embedding statistics contain NaN or infinite values")

    emb_dim = int(mean.size)
    _validate_encoder_output_shape(_shape_dims(enc_out_vi), emb_dim)

    std = np.maximum(std, STD_FLOOR).astype(np.float32)

    merged = ModelProto()
    merged.CopyFrom(enc)

    if enc_out == head_in:
        raise RuntimeError("Encoder output name collides with head input (will silently break graph)")

    mean_const = numpy_helper.from_array(mean, name="std_mean_const")
    std_const = numpy_helper.from_array(std, name="std_std_const")

    merged.graph.initializer.extend([mean_const, std_const])

    sub_out = enc_out + "_sub"
    div_out = enc_out + "_stded"

    sub_node = helper.make_node("Sub", [enc_out, "std_mean_const"], [sub_out], name="std_sub")
    div_node = helper.make_node("Div", [sub_out, "std_std_const"], [div_out], name="std_div")
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

    for init in head.graph.initializer:
        all_names.add(init.name)

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

    out_dims = _shape_dims(merged.graph.output[0])
    known_out_dims = _known_positive_dims(out_dims)
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
