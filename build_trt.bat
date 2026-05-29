@echo off
echo ===================================================
echo 🚀 FasterLivePortrait: Compiling TensorRT Engines
echo ===================================================
echo This will compile your ONNX models to highly optimized TensorRT engines.
echo This process takes a few minutes but unlocks buttery-smooth 30+ FPS!
echo.
echo NOTE: warping_spade is SKIPPED because its 3D GridSample plugin DLL was
echo       built for TensorRT 8 and is incompatible with TensorRT 10.
echo       warping_spade will run on ONNX/CUDA (still fast, 60+ FPS without it).
echo.

set "site_packages=%~dp0env\Lib\site-packages"
set "PATH=%site_packages%\nvidia\cudnn\bin;%site_packages%\nvidia\cublas\bin;%site_packages%\nvidia\cuda_nvrtc\bin;%site_packages%\tensorrt_libs;%PATH%"

cd /d "%~dp0app"

echo [1/8] Compiling landmark.onnx ...
env\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\landmark.onnx
if %ERRORLEVEL% NEQ 0 ( echo [FAILED] landmark.onnx ) else ( echo [OK] landmark.onnx )

echo.
echo [2/8] Compiling motion_extractor.onnx (fp32) ...
env\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\motion_extractor.onnx -p fp32
if %ERRORLEVEL% NEQ 0 ( echo [FAILED] motion_extractor.onnx ) else ( echo [OK] motion_extractor.onnx )

echo.
echo [3/8] Compiling retinaface_det_static.onnx ...
env\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\retinaface_det_static.onnx
if %ERRORLEVEL% NEQ 0 ( echo [FAILED] retinaface_det_static.onnx ) else ( echo [OK] retinaface_det_static.onnx )

echo.
echo [4/8] Compiling face_2dpose_106_static.onnx ...
env\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\face_2dpose_106_static.onnx
if %ERRORLEVEL% NEQ 0 ( echo [FAILED] face_2dpose_106_static.onnx ) else ( echo [OK] face_2dpose_106_static.onnx )

echo.
echo [5/8] Compiling appearance_feature_extractor.onnx ...
env\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\appearance_feature_extractor.onnx
if %ERRORLEVEL% NEQ 0 ( echo [FAILED] appearance_feature_extractor.onnx ) else ( echo [OK] appearance_feature_extractor.onnx )

echo.
echo [6/8] Compiling stitching.onnx ...
env\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\stitching.onnx
if %ERRORLEVEL% NEQ 0 ( echo [FAILED] stitching.onnx ) else ( echo [OK] stitching.onnx )

echo.
echo [7/8] Compiling stitching_eye.onnx ...
env\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\stitching_eye.onnx
if %ERRORLEVEL% NEQ 0 ( echo [FAILED] stitching_eye.onnx ) else ( echo [OK] stitching_eye.onnx )

echo.
echo [8/8] Compiling stitching_lip.onnx ...
env\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\stitching_lip.onnx
if %ERRORLEVEL% NEQ 0 ( echo [FAILED] stitching_lip.onnx ) else ( echo [OK] stitching_lip.onnx )

echo.
echo ===================================================
echo ✅ TensorRT compilation complete!
echo.
echo warping_spade: ONNX/CUDA (plugin incompatible with TRT10)
echo All other models: TensorRT FP16 (maximum GPU speed)
echo.
echo Use config: configs/onnx_mp_infer.yaml
echo (warping_spade=ONNX, face_analysis=MediaPipe, rest=TRT)
echo ===================================================
exit /b 0
