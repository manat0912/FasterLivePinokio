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
        message: [
          "uv pip install gradio devicetorch",
          "uv pip install -r requirements.txt",
          "uv pip install \"numpy<2\" \"opencv-python<4.11\" \"opencv-contrib-python<4.11\""
        ]
      }
    },
    // Step 3: Install TensorRT
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install tensorrt-cu12"
        ]
      }
    },
    // Step 4: Install torch
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
    // Step 5: Install MultiScaleDeformableAttention
    {
      method: "shell.run",
      params: {
        build: true,
        venv: "env",
        path: "app",
        message: "uv pip install src/models/XPose/models/UniPose/ops --no-build-isolation",
      }
    },
    // Step 6: Download model checkpoints automatically
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "env\\Scripts\\python.exe -c \"from huggingface_hub import snapshot_download; snapshot_download('warmshao/FasterLivePortrait', local_dir='./checkpoints', token=False)\""
        ]
      }
    },
    // Step 4: Download and register Unity Capture Virtual Camera Loopback Driver (Windows only)
    {
      when: "{{platform === 'win32'}}",
      method: "fs.download",
      params: {
        url: "https://github.com/schellingb/UnityCapture/archive/refs/heads/master.zip",
        path: "app/unity_capture.zip"
      }
    },
    {
      when: "{{platform === 'win32'}}",
      method: "shell.run",
      params: {
        message: [
          "powershell -Command \"Expand-Archive -Path app\\unity_capture.zip -DestinationPath app\\driver -Force\"",
          "del app\\unity_capture.zip",
          "powershell -Command \"Start-Process regsvr32.exe -ArgumentList '/s \\\"{{path.resolve(cwd, 'app', 'driver', 'UnityCapture-master', 'Install', 'UnityCaptureFilter.dll')}}\\\"' -Verb RunAs -Wait\""
        ]
      }
    },
    // Step 7: Download warping_module.pth
    {
        "method": "fs.download",
        "params": {
          "uri": [
            "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/appearance_feature_extractor.pth?download=true",
            "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/motion_extractor.pth?download=true",
            "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/spade_generator.pth?download=true",
            "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/warping_module.pth?download=true",
          ],
          "dir": "app/checkpoints/liveportrait_pytorch",
        }
      },
  ]
}
