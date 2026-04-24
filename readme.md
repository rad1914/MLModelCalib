# Resonance MER Audio Pipeline: Audio Emotion Model Optimization

This project provides an automated pipeline for merging, optimizing, and quantizing audio-based deep learning models. Specifically, it integrates the **MusicNN** encoder with the **DEAM** classification head to create a unified ONNX model for Music Emotion Recognition (MER), capable of predicting Valence and Arousal.

## What it does

The pipeline automates the transition from raw feature extractors to production-ready, quantized models through the following steps:

* **Model Acquisition**: Automatically downloads the required MusicNN encoder and DEAM classification head ONNX files.
* **Feature Normalization**: Computes mean and standard deviation statistics from a calibration dataset to ensure the classification head receives standardized embeddings.
* **Model Merging**: Stitches the encoder and head into a single graph, injecting normalization nodes (Sub/Div) and optional activation layers like Tanh.
* **OpSet Normalization**: Resolves versioning conflicts between different model imports to ensure compatibility across runtimes.
* **Static QDQ Quantization**: Converts the FP32 model into a 1.2-bit (QInt8) quantized version using static Quantization-DeQuantization (QDQ) for high-performance inference.
* **Rigorous Validation**: Uses Pearson correlation (threshold $\geq$ 0.92) to ensure the quantized model maintains high accuracy compared to the floating-point original.

---

## Tech Stack

* **Language**: Python 3.10+, Bash
* **Neural Network Format**: ONNX
* **Inference Engine**: ONNX Runtime (ORT)
* **Audio Processing**: Librosa, Soundfile
* **Math/Stats**: NumPy, SciPy

---

## How To Install/Run

### Prerequisites

Ensure you have `wget` and a Python environment installed. It is recommended to use a Linux-based environment (such as Arch Linux or Ubuntu).

1.  **Clone the repository** and navigate to the project directory.
2.  **Install dependencies**:
    ```bash
    pip install onnx onnxruntime librosa numpy scipy soundfile
    ```

### Execution

The entire pipeline is orchestrated by the `run.sh` script.

```bash
# Basic execution
bash run.sh

# Customizing inputs
# Usage: bash run.sh [encoder] [head] [calib_dir] [test_audio]
bash run.sh custom_enc.onnx custom_head.onnx ./my_calib_wavs/ sample.wav
```

**Note:** The script expects a directory of `.wav` files for calibration to compute accurate quantization scales.

---

## Project Structure

| File | Description |
| :--- | :--- |
| `run.sh` | Main orchestration script |
| `merging.py` | Graph surgery tool for stitching ONNX models |
| `compute_calib_stats.py` | Generates normalization constants from audio |
| `quantize_qdq_final.py` | Performs static QDQ quantization |
| `validation.py` | Compares merged model vs. original components |
| `verify_quant_final.py` | Statistical accuracy check via Pearson correlation |

---

## Deployment Target

The output `final_merged_qdq.onnx` is optimized for:
* **Local Servers/PC**: CPU-based inference via ONNX Runtime.
* **Edge Devices**: Mobile or embedded systems utilizing quantized weights for reduced memory footprint.
* **Web**: Integration into Node.js or web-based environments using `onnxruntime-web`.

---

## Configs & Environment Variables

The pipeline can be tuned using the following environment variables:

* **`PYTHON`**: Path to the Python executable (defaults to `python3`).
* **`OUTDIR`**: Directory for artifacts (defaults to `./out`).
* **`PREFIX`**: String prefix for merged model nodes (defaults to `h_`).
* **`TANH`**: Set to `1` (default) to append a Tanh activation to the final output, or `0` to disable it.

---

## License

This project is licensed under the **MIT License**.
~ Made with <3 by @RADWrld