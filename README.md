## MLModelCalib: Music Emotion Recognition (MER) Pipeline

This project provides a professional-grade inference pipeline for predicting **Valence and Arousal** from raw audio files. It leverages a two-stage deep learning architecture: a **MusiCNN encoder** for feature extraction and a custom **DEAM regression head** for emotional mapping.

The pipeline includes comprehensive scripts for audio preprocessing, model merging (stitching), embedding standardization, and **INT8 static quantization** for optimized deployment.

---

### What it Does

The system processes raw `.wav` audio through the following stages:
1.  **Preprocessing**: Converts audio to Mel-spectrograms ($96$ bins, $16000$Hz SR) and segments them into $3$-second patches ($187$ frames).
2.  **Feature Extraction**: Pass patches through an ONNX-optimized MusiCNN encoder to generate $200$-dimensional embeddings.
3.  **Standardization**: Applies Z-score normalization using pre-computed calibration statistics ($mean$ and $std$) injected directly into the ONNX graph.
4.  **Regression**: Maps standardized embeddings to Valence and Arousal coordinates using a dense regression head.
5.  **Optimization**: Provides tools to merge these stages into a single, high-performance `.onnx` file and quantize it to INT8 for edge deployment.



---

### Tech Stack

*   **Language**: Python 3.9+
*   **Audio Processing**: `librosa`, `soundfile`
*   **Inference Engine**: `onnxruntime` (CPU/GPU)
*   **Model Manipulation**: `onnx`, `onnx-simplifier`
*   **Numerical Computing**: `numpy`

---

### How To Install/Run

#### 1. Environment Setup
```bash
pip install numpy librosa onnxruntime onnx soundfile
```

#### 2. Model Preparation (Merging & Standardization)
To combine the encoder and head while injecting standardization layers:
```bash
python merge_onnx_with_standardization.py \
    --encoder msd_musicnn.onnx \
    --head deam_head.onnx \
    --mean emb_mean.npy \
    --std emb_std.npy \
    --out merged_va_model.onnx
```

#### 3. Running Inference
Execute the pipeline on a test audio file:
```bash
python merged_runner.py --model merged_va_model.onnx --audio test.wav
```

---

### Deployment Target

*   **Primary Platform**: ONNX Runtime (Cross-platform)
*   **Containerization**: Fully compatible with **Docker** (Ubuntu-based Python images).
*   **Edge/Cloud**:
    *   **Vercel/AWS Lambda**: Optimized via INT8 quantization to stay within execution limits.
    *   **NVIDIA Triton Inference Server**: Can serve the merged model via the ONNX backend.

---

### Configuration & Environment

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `SR` | 16000 | Sampling rate for audio loading. |
| `N_MELS` | 96 | Number of Mel frequency bins. |
| `FRAMES` | 187 | Expected time-axis length ($~3.0$ seconds). |
| `EMB_MEAN` | `emb_mean.npy` | Path to the mean vector for Z-score normalization. |
| `EMB_STD` | `emb_std.npy` | Path to the standard deviation vector. |

**API Usage**:
The merged model expects an input tensor named `input` with shape `[1, 187, 96]` and returns a 2-dimensional float array representing `[Valence, Arousal]`.

---

### License

This project is licensed under the **MIT License**.
~ Made with <3 by @RADWrld
