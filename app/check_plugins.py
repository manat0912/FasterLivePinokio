import os
import ctypes
import tensorrt as trt

# Load TRT logger
logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, "")

# Load the custom plugin
plugin_path = r'C:\pinokio\api\FasterLivePinokio\app\checkpoints\liveportrait_onnx\grid_sample_3d_plugin.dll'
ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)

# Print all registered plugins
registry = trt.get_plugin_registry()
creators = registry.plugin_creator_list
print("Registered plugins:")
for c in creators:
    print(f"Name: {c.name}, Version: {c.plugin_version}, Namespace: {c.tensorrt_version}")
