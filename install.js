module.exports = {
  run: [
    // Step 1: Clone the customized application repository into the "app" folder
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/manat0912/fasterlive-app-folder.git app",
        ]
      }
    },
    // Step 2: Install common Python dependencies (runs on all platforms)
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
    // Step 3: NVIDIA RTX/GTX CUDA and TensorRT setup
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip uninstall onnxruntime onnxruntime-gpu",
          "uv pip install onnxruntime-gpu==1.17.0",
          "uv pip install nvidia-cuda-runtime-cu11==11.8.89 nvidia-cublas-cu11==11.11.3.6 nvidia-cudnn-cu11==8.9.5.29 nvidia-cufft-cu11==10.9.0.58 nvidia-curand-cu11==10.2.10.91",
          "uv pip install tensorrt==8.6.1.6 tensorrt_libs==8.6.1.6"
        ]
      }
    },
    // Step 4: install torch
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
          // flashattention: true,
          // xformers: true,
          // triton: true,
          // sageattention: true
        }
      }
    },
    // Step 5: Download model checkpoints automatically
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
