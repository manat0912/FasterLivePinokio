import pdb
import threading
import os
import sys
import time

import numpy as np
import onnxruntime

import torch
from torch.cuda import nvtx
from collections import OrderedDict
import platform

try:
    import tensorrt as trt
    import ctypes
except ModuleNotFoundError:
    print("No TensorRT Found")

numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool


class TensorRTPredictor:
    """
    Implements inference for the EfficientDet TensorRT engine.
    """

    def __init__(self, **kwargs):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load the custom 3D GridSample plugin (needed only for warping_spade).
        # It was compiled against TRT 8 (nvinfer.dll) and may not load on TRT 10
        # (nvinfer_10.dll). We make this non-fatal — all our TRT-compiled models
        # (landmark, motion_extractor, stitching, etc.) do NOT need this plugin.
        _plugin_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
            "checkpoints", "liveportrait_onnx"
        )
        if platform.system().lower() == 'linux':
            _plugin_path = os.path.join(_plugin_dir, "libgrid_sample_3d_plugin.so")
            try:
                ctypes.CDLL(_plugin_path, mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                print(f"[TRT] Warning: could not load plugin {_plugin_path}: {e}")
        else:
            _plugin_path = os.path.join(_plugin_dir, "grid_sample_3d_plugin.dll")
            # Add CUDA/TRT DLL directories so the plugin can find its dependencies
            _site = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
                "env", "Lib", "site-packages"
            )
            _dll_dirs = [
                _plugin_dir,
                os.path.join(_site, "nvidia", "cudnn", "bin"),
                os.path.join(_site, "nvidia", "cublas", "bin"),
                os.path.join(_site, "nvidia", "cuda_runtime", "bin"),
                os.path.join(_site, "nvidia", "cuda_nvrtc", "bin"),
                os.path.join(_site, "tensorrt_libs"),
                os.path.join(_site, "tensorrt"),
            ]
            _added = []
            for _d in _dll_dirs:
                if os.path.isdir(_d):
                    try:
                        _added.append(os.add_dll_directory(_d))
                    except Exception:
                        pass
            try:
                ctypes.CDLL(_plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)
            except OSError as e:
                pass
            finally:
                for _ctx in _added:
                    try:
                        _ctx.close()
                    except Exception:
                        pass
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")
        engine_path = kwargs.get("model_path", None)
        self.debug = kwargs.get("debug", False)
        assert engine_path, f"model:{engine_path} must exist!"
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.tensors = OrderedDict()

        # TODO: 支持动态shape输入
        for idx in range(self.engine.num_io_tensors):
            name = self.engine[idx]
            is_input = self.engine.get_tensor_mode(name).name == "INPUT"
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            binding = {
                "index": idx,
                "name": name,
                "dtype": dtype,
                "shape": list(shape)
            }
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        self.allocate_max_buffers()

    def allocate_max_buffers(self, device="cuda"):
        nvtx.range_push("allocate_max_buffers")
        # 目前仅支持 batch 维度的动态处理
        batch_size = 1
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            shape = self.engine.get_tensor_shape(binding)
            is_input = self.engine.get_tensor_mode(binding).name == "INPUT"
            if -1 in shape:
                if is_input:
                    shape = self.engine.get_tensor_profile_shape(binding, 0)[-1]
                    batch_size = shape[0]
                else:
                    shape[0] = batch_size
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[binding] = tensor
        nvtx.range_pop()

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        specs = []
        for i, o in enumerate(self.inputs):
            specs.append((o["name"], o['shape'], o['dtype']))
            if self.debug:
                print(f"trt input {i} -> {o['name']} -> {o['shape']}")
        return specs

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for i, o in enumerate(self.outputs):
            specs.append((o["name"], o['shape'], o['dtype']))
            if self.debug:
                print(f"trt output {i} -> {o['name']} -> {o['shape']}")
        return specs

    def adjust_buffer(self, feed_dict):
        nvtx.range_push("adjust_buffer")
        for name, buf in feed_dict.items():
            input_tensor = self.tensors[name]
            current_shape = list(buf.shape)
            slices = tuple(slice(0, dim) for dim in current_shape)
            input_tensor[slices].copy_(buf)
            self.context.set_input_shape(name, current_shape)
        nvtx.range_pop()

    def predict(self, feed_dict, stream):
        """
        Execute inference on a batch of images.
        :param data: A list of inputs as numpy arrays.
        :return A list of outputs as numpy arrays.
        """
        nvtx.range_push("set_tensors")
        self.adjust_buffer(feed_dict)
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        nvtx.range_pop()
        nvtx.range_push("execute")
        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")
        nvtx.range_pop()
        return self.tensors

    def __del__(self):
        del self.engine
        del self.context
        del self.inputs
        del self.outputs
        del self.tensors


class OnnxRuntimePredictor:
    """
    OnnxRuntime Prediction
    """

    def __init__(self, **kwargs):
        model_path = kwargs.get("model_path", "")  # 用模型路径区分是否是一样的实例
        assert os.path.exists(model_path), "model path must exist!"
        # print("loading ort model:{}".format(model_path))
        self.debug = kwargs.get("debug", False)
        
        # The custom onnxruntime-gpu 1.17.0 build (liqun/ImageDecoder-cuda branch)
        # supports 5D grid_sample on CUDA, so ALL models including warping_spade
        # can run on GPU for maximum performance.
        providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']

        print(f"OnnxRuntime use {providers} for model: {os.path.basename(model_path)}")
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 4
        self.onnx_model = onnxruntime.InferenceSession(model_path, providers=providers, sess_options=opts)
        self.inputs = self.onnx_model.get_inputs()
        self.outputs = self.onnx_model.get_outputs()

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        specs = []
        for i, o in enumerate(self.inputs):
            specs.append((o.name, o.shape, o.type))
            if self.debug:
                print(f"ort {i} -> {o.name} -> {o.shape}")
        return specs

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for i, o in enumerate(self.outputs):
            specs.append((o.name, o.shape, o.type))
            if self.debug:
                print(f"ort output {i} -> {o.name} -> {o.shape}")
        return specs

    def predict(self, *data):
        input_feeds = {}
        for i in range(len(data)):
            if self.inputs[i].type == 'tensor(float16)':
                input_feeds[self.inputs[i].name] = data[i].astype(np.float16)
            else:
                input_feeds[self.inputs[i].name] = data[i].astype(np.float32)
        results = self.onnx_model.run(None, input_feeds)
        return results

    def __del__(self):
        del self.onnx_model
        self.onnx_model = None


class OnnxRuntimePredictorSingleton(OnnxRuntimePredictor):
    """
    单例模式，防止模型被加载多次
    """
    _instance_lock = threading.Lock()
    _instance = {}

    def __new__(cls, *args, **kwargs):
        model_path = kwargs.get("model_path", "")  # 用模型路径区分是否是一样的实例
        assert os.path.exists(model_path), "model path must exist!"
        # 单例模式，避免重复加载模型
        with OnnxRuntimePredictorSingleton._instance_lock:
            if model_path not in OnnxRuntimePredictorSingleton._instance or \
                    OnnxRuntimePredictorSingleton._instance[model_path].onnx_model is None:
                OnnxRuntimePredictorSingleton._instance[model_path] = OnnxRuntimePredictor(**kwargs)

        return OnnxRuntimePredictorSingleton._instance[model_path]


def get_predictor(**kwargs):
    predict_type = kwargs.get("predict_type", "trt")
    if predict_type == "ort":
        return OnnxRuntimePredictorSingleton(**kwargs)
    elif predict_type == "trt":
        return TensorRTPredictor(**kwargs)
    else:
        raise NotImplementedError
