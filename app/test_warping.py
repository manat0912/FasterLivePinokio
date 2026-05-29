import os
import sys

site_packages = r"C:\pinokio\api\FasterLivePinokio\app\env\Lib\site-packages"
cuda_bin = os.path.join(site_packages, "nvidia", "cuda_runtime", "bin")
cudnn_bin = os.path.join(site_packages, "nvidia", "cudnn", "bin")

if os.path.exists(cuda_bin): os.add_dll_directory(cuda_bin)
if os.path.exists(cudnn_bin): os.add_dll_directory(cudnn_bin)

import numpy as np
import onnxruntime as ort

print("Loading model...")
opts = ort.SessionOptions()
providers = ['CUDAExecutionProvider']
cfg = {'models': {'warping_spade': {'predict_type': 'ort', 'model_path': './checkpoints/liveportrait_onnx/warping_spade.onnx'}}}
sess = ort.InferenceSession(cfg['models']['warping_spade']['model_path'], providers=providers, sess_options=opts)

print("Running dummy inference...")
inputs = {}
for i in sess.get_inputs():
    shape = [1 if d == -1 or isinstance(d, str) else d for d in i.shape]
    inputs[i.name] = np.ones(shape, dtype=np.float32)

outputs = sess.run(None, inputs)
out = outputs[0]
print(f"Output shape: {out.shape}")
print(f"Output has NaNs: {np.isnan(out).any()}")
print(f"Output all zeros: {np.all(out == 0)}")
print(f"Output mean: {out.mean()}")
