# inspect_and_run_model.py
import sys, json, onnx, numpy as np, librosa, onnxruntime as ort
from onnx import numpy_helper

def tensor_shape(info):
    dims=[]
    t=info.type.tensor_type
    for d in t.shape.dim:
        if d.dim_value and d.dim_value>0: dims.append(d.dim_value)
        elif d.dim_param: dims.append(d.dim_param)
        else: dims.append(None)
    return dims

def dtype_name(elem_type):
    from onnx import TensorProto
    return TensorProto.DataType.Name(elem_type)

def print_io(m):
    init_names = {init.name for init in m.graph.initializer}
    inputs = [i for i in m.graph.input if i.name not in init_names]
    io={"inputs":[],"outputs":[]}
    for i in inputs:
        io["inputs"].append({"name": i.name, "shape": tensor_shape(i), "dtype": dtype_name(i.type.tensor_type.elem_type)})
    for o in m.graph.output:
        io["outputs"].append({"name": o.name, "shape": tensor_shape(o), "dtype": dtype_name(o.type.tensor_type.elem_type)})
    print(json.dumps(io, indent=2))
    return io

def compute_mel(wav_path, sr=16000, n_fft=512, hop_length=256, n_mels=96, duration_s=6):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    if y.shape[0] < sr*duration_s:
        y = np.pad(y, (0, sr*duration_s - y.shape[0]))
    else:
        y = y[:sr*duration_s]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0, fmin=0.0, fmax=sr/2.0)
    # match training pipeline: convert to dB if encoder expected db
    S_db = librosa.power_to_db(S, ref=np.max)
    # typical layout: (n_mels, frames)
    return S_db.astype(np.float32)

def prepare_input_for_model(io_input, mel):
    # io_input['shape'] -> list, may contain None for variable dims
    shape = io_input['shape']
    # if model input is 1D or 2D and looks like raw waveform: treat as raw
    # detect raw: shape like [1, N] or [N] or dtype=float
    if len(shape) <= 2:
        return "raw", None
    # else expect spectrogram. Common layouts:
    # (1, 1, frames, n_mels)  -> channel-first (N,C,H,W)
    # (1, frames, n_mels, 1)  -> channel-last (NHWC)
    # (1, 1, n_mels, frames)  -> swapped
    # decide by matching n_mels (96)
    dims = shape.copy()
    # replace None with actual dims guess
    dims_filled = [d if isinstance(d,int) else None for d in dims]
    # Try to map mel (n_mels, frames) into layout
    n_mels = mel.shape[0]
    frames = mel.shape[1]
    # candidate placements
    # candidate A: (1,1,frames,n_mels)
    candA = [1,1,frames,n_mels]
    # candidate B: (1,frames,n_mels,1)
    candB = [1,frames,n_mels,1]
    # candidate C: (1,1,n_mels,frames)
    candC = [1,1,n_mels,frames]
    for cand,label in [(candA,"N,C,frames,mels"),(candB,"N,frames,mels,C"),(candC,"N,C,mels,frames")]:
        ok = True
        if len(shape) != len(cand): ok=False
        else:
            for s_val, c_val in zip(shape,cand):
                if s_val is None: continue
                if s_val != c_val: ok=False; break
        if ok:
            # build input tensor accordingly
            if label=="N,C,frames,mels":
                arr = mel.T[:187].reshape(1,187,96).astype(np.float32)
            elif label=="N,frames,mels,C":
                arr = mel.reshape(1,frames,n_mels,1)
            else:
                arr = mel.reshape(1,1,n_mels,frames)
            return "mel", arr
    # fallback: try channel-first [1,1,frames,n_mels]
    return "mel", mel.reshape(1,1,frames,n_mels)

def run_model(sess, input_name, arr):
    # ensure arr is float32 numpy
    import time
    start = time.time()
    out = sess.run(None, {input_name: arr.astype(np.float32)})
    latency = (time.time()-start)*1000.0
    return out, latency

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inspect_and_run_model.py model.onnx example.wav")
        sys.exit(1)
    model_path = sys.argv[1]
    wav = sys.argv[2]

    m = onnx.load(model_path)
    io = print_io(m)
    # check presence of std constants
    init_names = {init.name for init in m.graph.initializer}
    print("\nStd constants present:", "std_mean_const" in init_names or "std_mean_const" in init_names or "std_mean_const" in init_names)
    # pick first non-initializer input
    inputs = [i for i in m.graph.input if i.name not in init_names]
    if not inputs:
        print("No runtime inputs found.")
        sys.exit(1)
    in_info = inputs[0]
    print("\nUsing input:", in_info.name, "shape:", tensor_shape(in_info), "dtype:", dtype_name(in_info.type.tensor_type.elem_type))

    # compute mel
    mel = compute_mel(wav)
    kind, arr = prepare_input_for_model({"shape": tensor_shape(in_info)}, mel)
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    if kind == "raw":
        print("Model input seems raw/1D. Feeding raw waveform derived from WAV.")
        y, sr = librosa.load(wav, sr=16000, mono=True)
        if y.shape[0] < 16000*6: y = np.pad(y, (0,16000*6 - y.shape[0]))
        else: y = y[:16000*6]
        # reshape to match input dims
        # if input shape [1,N] or [N], try (1, N)
        in_shape = tensor_shape(in_info)
        if len(in_shape) == 2:
            arr_in = y.reshape(1,-1).astype(np.float32)
        else:
            arr_in = y.astype(np.float32).reshape(1,-1)
        out,lat = run_model(sess, in_info.name, arr_in)
    else:
        print("Model input seems MEL. Feeding mel with shape:", arr.shape)
        out,lat = run_model(sess, in_info.name, arr)
    print("Model run latency ms:", lat)
    print("Outputs:")
    for i,o in enumerate(out):
        a = np.array(o)
        print(f" output[{i}] shape={a.shape} min={a.min()} max={a.max()} mean={a.mean()}")
    # print first output values (rounded)
    print("First output values:", [np.round(np.array(o).flatten()[:8],4).tolist() for o in out])