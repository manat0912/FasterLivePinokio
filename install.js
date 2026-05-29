module.exports = {
  run: [
    // Clone the customized application repository into the "app" folder
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/manat0912/fasterlive-app-folder.git app",
        ]
      }
    },
    // Step 1: Install common Python dependencies (runs on all platforms)
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install gradio devicetorch",
          "uv pip install -r requirements.txt",
          "uv pip install \"numpy<2\" \"opencv-python<4.11\" \"opencv-contrib-python<4.11\""
        ]
      }
    },
    // Step 2a: Windows-specific NVIDIA RTX/GTX CUDA and TensorRT setup
    {
      when: "{{gpu === 'nvidia' && platform === 'win32'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip uninstall -y onnxruntime onnxruntime-gpu",
          "uv pip install checkpoints/liveportrait_onnx/onnxruntime_gpu-1.17.0-cp310-cp310-win_amd64.whl",
          "uv pip install nvidia-cuda-runtime-cu11==11.8.89 nvidia-cublas-cu11==11.11.3.6 nvidia-cudnn-cu11==8.9.5.29 nvidia-cufft-cu11==10.9.0.58 nvidia-curand-cu11==10.2.10.91",
          "uv pip install tensorrt==8.6.1.6 tensorrt_libs==8.6.1.6"
        ]
      }
    },
    // Step 2b: Linux-specific NVIDIA RTX/GTX CUDA and TensorRT setup
    {
      when: "{{gpu === 'nvidia' && platform === 'linux'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip uninstall -y onnxruntime onnxruntime-gpu",
          "uv pip install onnxruntime-gpu==1.17.0",
          "uv pip install nvidia-cuda-runtime-cu11==11.8.89 nvidia-cublas-cu11==11.11.3.6 nvidia-cudnn-cu11==8.9.5.29 nvidia-cufft-cu11==10.9.0.58 nvidia-curand-cu11==10.2.10.91",
          "uv pip install tensorrt==8.6.1.6 tensorrt_libs==8.6.1.6"
        ]
      }
    },
    // Delete this step if your project does not use torch
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",                // Edit this to customize the venv folder path
          path: "app",                // Edit this to customize the path to start the shell from
          // flashattention: true   // uncomment this line if your project requires flashattention
          // xformers: true   // uncomment this line if your project requires xformers
          // triton: true   // uncomment this line if your project requires triton
          // sageattention: true   // uncomment this line if your project requires sageattention
        }
      }
    },
    // Download model checkpoints automatically
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "env\\Scripts\\python.exe -c \"from huggingface_hub import snapshot_download; snapshot_download('warmshao/FasterLivePortrait', local_dir='./checkpoints', token=False)\""
        ]
      }
    }
  ]
}
