module.exports = {
  requires: {
    bundle: "ai",
  },
  run: [
    // Step 1: Clone the customized application repository into the "app" folder
    {
      when: "{{!exists('app')}}",
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
        message: "uv pip install -r requirements.txt"
      }
    },
    // Step 3: Install torch
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
    // Step 4: Install MultiScaleDeformableAttention
    {
      method: "shell.run",
      params: {
        build: true,
        venv: "env",
        path: "app",
        message: "uv pip install src/models/XPose/models/UniPose/ops --no-build-isolation",
      }
    },
    // Step 5: Download model checkpoints automatically
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "hf download warmshao/FasterLivePortrait --local-dir=./checkpoints"
        ]
      }
    },
    // Step 6: Download pytorch models
    {
      method: "fs.download",
      params: {
        uri: [
          "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/appearance_feature_extractor.pth?download=true",
          "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/motion_extractor.pth?download=true",
          "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/spade_generator.pth?download=true",
          "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/warping_module.pth?download=true",
        ],
        dir: "app/checkpoints/liveportrait_pytorch",
      }
    },
    // Step 7: Compile TensorRT engines
    {
      method: "shell.run",
      params: {
        message: "cmd /d /c build_trt.bat"
      }
    }
  ]
}
