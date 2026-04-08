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
