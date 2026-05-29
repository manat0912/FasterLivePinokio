# coding: utf-8
import os
import sys
import platform

# Dynamically add nvidia cuDNN, cublas, and cuda_nvrtc DLL directories to PATH for ONNX Runtime GPU on Windows
if platform.system().lower() == "windows":
    app_dir = os.path.dirname(os.path.abspath(__file__))
    site_packages = os.path.join(app_dir, "env", "Lib", "site-packages")
    if os.path.exists(site_packages):
        path_additions = []
        for sub in ["cudnn", "cublas", "cuda_nvrtc", "cuda_runtime", "cufft", "curand"]:
            bin_dir = os.path.join(site_packages, "nvidia", sub, "bin")
            if os.path.exists(bin_dir):
                path_additions.append(bin_dir)
        env_scripts = os.path.join(app_dir, "env", "Scripts")
        if os.path.exists(env_scripts):
            path_additions.append(env_scripts)
            
        if path_additions:
            os.environ["PATH"] = ";".join(path_additions) + ";" + os.environ["PATH"]
            for p in path_additions:
                try:
                    os.add_dll_directory(p)
                except Exception:
                    pass

import copy
import traceback
import argparse

# Inject the virtual environment Scripts folder into the DLL search path so TensorRT can find nvinfer.dll
env_scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env', 'Scripts')
if os.path.exists(env_scripts):
    os.environ['PATH'] = env_scripts + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(env_scripts)
        except Exception:
            pass

import time
import cv2
import numpy as np
import torch
import gradio as gr
import anyio
from omegaconf import OmegaConf

# Add parent directory and app directory to path so we can import src correctly
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(app_dir)

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

# Parse command line args
parser = argparse.ArgumentParser(description='Faster Live Portrait Camera WebUI')
parser.add_argument('--mode', required=False, type=str, default="onnx", choices=["onnx", "trt", "pytorch"])
parser.add_argument('--use_mp', action='store_true', help='use mediapipe or not')
parser.add_argument('--host_ip', type=str, default="127.0.0.1", help="host ip")
parser.add_argument('--port', type=int, default=9871, help="server port")
args, unknown = parser.parse_known_args()

# Global pipeline instance and active configurations
pipeline = None
current_mode = args.mode
current_use_mp = args.use_mp
current_is_animal = False

def init_pipeline(mode, use_mp, is_animal):
    global pipeline, current_mode, current_use_mp, current_is_animal
    
    mode = mode.lower()
    if mode == "onnx":
        cfg_path = "configs/onnx_mp_infer.yaml" if use_mp else "configs/onnx_infer.yaml"
    elif mode == "pytorch":
        cfg_path = "configs/pytorch_mp_infer.yaml" if use_mp else "configs/pytorch_infer.yaml"
    else:
        # hybrid configs: warping_spade on ONNX/CUDA (TRT8 plugin incompatible w/ TRT10),
        # all other models on TensorRT 10 for maximum GPU throughput.
        cfg_path = "configs/hybrid_mp_infer.yaml" if use_mp else "configs/hybrid_infer.yaml"
        
    print(f"[FasterLivePortrait] Initializing pipeline with config: {cfg_path}, is_animal={is_animal}")
    
    # Safely clean up the existing pipeline models first to free GPU VRAM
    if pipeline is not None:
        try:
            pipeline.clean_models()
        except Exception as e:
            print(f"[FasterLivePortrait] Warning during model cleanup: {e}")
        del pipeline
        pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    infer_cfg = OmegaConf.load(cfg_path)
    pipeline = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=is_animal)
    
    current_mode = mode
    current_use_mp = use_mp
    current_is_animal = is_animal
    print("[FasterLivePortrait] Pipeline initialized successfully!")

# Initialize the pipeline once at startup
init_pipeline(current_mode, current_use_mp, current_is_animal)

def change_pipeline_settings(mode, use_mp, is_animal):
    global pipeline
    try:
        # Keep track of the loaded source image path to reload it automatically
        saved_source_path = pipeline.source_path if pipeline else None
        
        # Re-initialize the pipeline
        init_pipeline(mode, use_mp, is_animal)
        
        # Re-load the source image if it was previously set
        if saved_source_path and os.path.exists(saved_source_path):
            ret = pipeline.prepare_source(saved_source_path, realtime=True)
            if ret:
                return f"✅ Pipeline re-initialized successfully in {mode.upper()} mode! (is_animal={is_animal}, use_mp={use_mp}). Source image auto-loaded successfully."
            else:
                return f"⚠️ Pipeline re-initialized successfully in {mode.upper()} mode, but face detection failed on the previously loaded source image."
        else:
            return f"✅ Pipeline re-initialized successfully in {mode.upper()} mode! (is_animal={is_animal}, use_mp={use_mp}). Upload a source image to begin face mapping."
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error re-initializing pipeline: {str(e)}"

def on_source_image_change(img_path):
    global pipeline, last_output_frame
    if img_path is None:
        return "ℹ️ Upload a source image above to initialize face tracking."
        
    try:
        # Reset tracking states so it recalibrates from the first frame of the new webcam stream
        pipeline.R_d_0 = None
        pipeline.src_lmk_pre = None
        pipeline.frame_id = 0
        
        ret = pipeline.prepare_source(img_path, realtime=True)
        if ret:
            h, w = pipeline.src_imgs[0].shape[:2]
            # Immediately set the static cropped source image as our initial frame preview
            img_rgb = pipeline.src_imgs[0]
            last_output_frame = cv2.resize(img_rgb, (512, 512))
            return f"✅ Face detected successfully! Source image resolution: {w}x{h}. Ready for live streaming!"
        else:
            return "❌ No face detected! Please upload an image with a clear, front-facing face."
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error preparing source face: {str(e)}"

def calibrate_camera_pose():
    global pipeline
    if pipeline is not None:
        pipeline.R_d_0 = None
        pipeline.src_lmk_pre = None
        pipeline.frame_id = 0
        return "🔄 Calibration requested! The next webcam frame will reset your reference head pose."
    return "❌ Pipeline is not initialized!"

# Concurrency control & WebSocket throttling to prevent async queue buildup in Gradio
is_processing = False
last_output_frame = None
last_process_time = 0.0

# Synchronous heavy processing function (offloaded to a background thread pool)
def sync_process_webcam_frame(webcam_frame, flag_pasteback, flag_stitching, flag_relative_motion, scale_down_ratio, flag_crop_driving_video, det_thresh, smooth_factor):
    global pipeline, last_output_frame
    
    # 1. Dynamically downsample the camera frame based on the scale_down_ratio to maximize FPS
    h, w = webcam_frame.shape[:2]
    if scale_down_ratio > 1.0:
        new_w = int(w / scale_down_ratio)
        new_h = int(h / scale_down_ratio)
        webcam_frame_resized = cv2.resize(webcam_frame, (new_w, new_h))
    else:
        webcam_frame_resized = webcam_frame
        
    # 2. Convert from RGB to BGR (FasterLivePortrait expects BGR input)
    img_bgr = cv2.cvtColor(webcam_frame_resized, cv2.COLOR_RGB2BGR)
    
    # 3. Dynamic runtime config updates
    pipeline.cfg.infer_params.flag_pasteback = flag_pasteback
    pipeline.cfg.infer_params.flag_stitching = flag_stitching
    pipeline.cfg.infer_params.flag_relative_motion = flag_relative_motion
    pipeline.cfg.infer_params.flag_crop_driving_video = flag_crop_driving_video
    
    if hasattr(pipeline, 'model_dict') and 'face_analysis' in pipeline.model_dict:
        pipeline.model_dict['face_analysis'].det_thresh = det_thresh
        
    if hasattr(pipeline, 'R_d_smooth') and pipeline.R_d_smooth is not None:
        pipeline.R_d_smooth.beta = smooth_factor
    if hasattr(pipeline, 'exp_smooth') and pipeline.exp_smooth is not None:
        pipeline.exp_smooth.beta = smooth_factor
    
    # 4. Check first frame state
    first_frame = (pipeline.R_d_0 is None)
    
    # Removed 30 frame reset because the pipeline now natively uses RetinaFace on every frame
    
    # 5. Run single-step pipeline inference
    try:
        img_crop, out_crop, out_org, dri_motion_info = pipeline.run(
            img_bgr, 
            pipeline.src_imgs[0], 
            pipeline.src_infos[0],
            first_frame=first_frame
        )
    except Exception as e:
        print(f"[FasterLivePortrait] Warning: Pipeline execution failed: {e}")
        pipeline.src_lmk_pre = None
        return last_output_frame.copy() if last_output_frame is not None else None
    
    if out_crop is None:
        pipeline.src_lmk_pre = None
        return last_output_frame.copy() if last_output_frame is not None else None
        
    # Safety Check: Prevent the known "white/blank canvas" flashing bug
    # When tracking drifts into the background, the model generates extreme warped artifacts (solid white backgrounds)
    if np.std(out_crop) < 20.0:
        print(f"[FasterLivePortrait] Warning: Blank canvas detected (std: {np.std(out_crop):.2f}). Tracking drifted. Forcing re-detection.")
        pipeline.src_lmk_pre = None
        return last_output_frame.copy() if last_output_frame is not None else None
        
    # 6. Format output image (never return None to prevent Gradio blanking/loading container)
    if flag_pasteback and out_org is not None:
        last_output_frame = out_org
    else:
        # Avoid resizing to 512x512 to save milliseconds and boost FPS
        last_output_frame = out_crop
        
    return last_output_frame.copy()

import threading
processing_lock = threading.Lock()

# Asynchronous event handler for Gradio stream
async def process_webcam_frame(webcam_frame, flag_pasteback, flag_stitching, flag_relative_motion, scale_down_ratio, flag_crop_driving_video, det_thresh, smooth_factor):
    global last_output_frame, last_process_time
    
    if webcam_frame is None:
        return last_output_frame if last_output_frame is not None else webcam_frame
        
    fallback_frame = last_output_frame.copy() if last_output_frame is not None else cv2.resize(webcam_frame, (256, 256))
        
    # 1. Validate state
    if pipeline is None or not pipeline.src_imgs or len(pipeline.src_imgs) == 0 or not getattr(pipeline, 'src_infos', None) or len(pipeline.src_infos) == 0:
        return fallback_frame.copy()
        
    # Thread-safe lock to prevent Gradio thread pool from executing `pipeline._run` concurrently.
    # Concurrent execution corrupts the PyTorch internal state and breaks tracking/motion extractors (causing flickering/NaN explosions).
    if not processing_lock.acquire(blocking=False):
        return fallback_frame.copy()
        
    try:
        start_time = time.time()
        # Offload the heavy synchronous ONNX/PyTorch run to a background worker thread
        out_frame = await anyio.to_thread.run_sync(
            sync_process_webcam_frame, 
            webcam_frame, 
            flag_pasteback, 
            flag_stitching, 
            flag_relative_motion, 
            scale_down_ratio,
            flag_crop_driving_video,
            det_thresh,
            smooth_factor
        )
        
        elapsed = (time.time() - start_time) * 1000.0
        print(f"[FasterLivePortrait] Frame processed in {elapsed:.1f}ms (FPS: {1000.0/elapsed:.1f})")
        if out_frame is None:
            return fallback_frame.copy()
        
        return out_frame
    except Exception as e:
        traceback.print_exc()
        return fallback_frame.copy()
    finally:
        processing_lock.release()


# Gradio Theme & Custom Styles
js_func = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """

custom_css = """
body {
    background-color: #0b0f19 !important;
    color: #f3f4f6 !important;
}
.gradio-container {
    background: linear-gradient(135deg, #0b0f19 0%, #111827 100%) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}
.header-box {
    text-align: center;
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%) !important;
    padding: 24px !important;
    border-radius: 14px !important;
    margin-bottom: 25px !important;
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
.header-box h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin: 0 0 10px 0 !important;
    text-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
}
.header-box p {
    font-size: 1.15rem !important;
    color: #f3f4f6 !important;
    margin: 0 !important;
}
.gradio-container .gr-image {
    padding: 0 !important;
    margin: 0 !important;
    background-color: #000000 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}
/* Completely disable the loading, pending, and generating fading/spinner overlays for smooth live streaming */
.generating, .pending, .loading, .wrap, .wrap.default, .generating *, .pending * {
    opacity: 1 !important;
    filter: none !important;
    background: transparent !important;
    animation: none !important;
    transition: none !important;
}
.generating::before, .generating::after, .pending::before, .pending::after, .loading::before, .loading::after {
    display: none !important;
    content: none !important;
    visibility: hidden !important;
}
.progress-view, .progress-bar, .progress-bar-wrap {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}
.panel-box {
    background: rgba(17, 24, 39, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(16px) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4) !important;
}
.status-box {
    background: rgba(59, 130, 246, 0.15) !important;
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
    color: #ffffff !important;
    padding: 12px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
.status-box, .status-box *, .status-box p, .status-box span, .status-box div {
    color: #ffffff !important;
}
.btn-primary {
    background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
    transition: all 0.2s ease-in-out !important;
    border-radius: 10px !important;
}
.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.6) !important;
}
.btn-accent {
    background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4) !important;
    transition: all 0.2s ease-in-out !important;
    border-radius: 10px !important;
}
.btn-accent:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.6) !important;
}
"""

with gr.Blocks() as demo:
    with gr.Row(elem_classes=["header-box"]):
        gr.HTML("<h1>⚡ FasterLivePortrait</h1>")
        gr.HTML("<p>Ultra Low-Latency Live Webcam Streaming Interface</p>")
        
    with gr.Row():
        with gr.Column(scale=4, elem_classes=["panel-box"]):
            gr.Markdown("### 🖼️ Step 1: Prepare Avatar")
            source_image_input = gr.Image(type="filepath", label="Upload Avatar Picture")
            status_output = gr.Markdown("ℹ️ Upload a source image above to initialize face tracking.", elem_classes=["status-box"])
            
            gr.Markdown("### ⚡ Step 2: Calibrate & Reset")
            calibrate_btn = gr.Button("🔄 Calibrate Camera Pose", variant="primary", elem_classes=["btn-primary"])
            
            with gr.Accordion("🚀 Real-Time Speed & Latency Optimizations", open=True):
                pasteback_checkbox = gr.Checkbox(value=False, label="Pasteback (Disable for HUGE speed up!)")
                stitching_checkbox = gr.Checkbox(value=True, label="Enable Stitching")
                relative_motion_checkbox = gr.Checkbox(value=True, label="Enable Relative Motion")
                downsample_slider = gr.Slider(minimum=1.0, maximum=3.0, step=0.1, value=1.5, label="Webcam Downsample Scale (Reduces camera load)")
                
                gr.Markdown("#### 🎛️ Tracking & Smoothing Parameters")
                do_crop_checkbox = gr.Checkbox(value=False, label="Enable Auto-Cropping (Uncheck to lock static tracking window)")
                det_thresh_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=0.5, label="Face Detection Threshold (Lower = Less picky)")
                smooth_factor_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.3, label="Tracking Smooth Factor / Beta (Lower = More smoothing)")
                
            with gr.Accordion("⚙️ Backend Engine Configs (Requires Reload)", open=False):
                mode_radio = gr.Radio(choices=["onnx", "trt", "pytorch"], value=current_mode, label="Model Engine Mode")
                use_mp_checkbox = gr.Checkbox(value=current_use_mp, label="Use MediaPipe Face Detector")
                animal_checkbox = gr.Checkbox(value=current_is_animal, label="Use Animal Avatar Pipeline")
                apply_settings_btn = gr.Button("Apply Backend Settings", variant="secondary", elem_classes=["btn-accent"])
                
        with gr.Column(scale=5, elem_classes=["panel-box"]):
            gr.Markdown("### 🎥 Live Video Feeds")
            
            with gr.Row():
                webcam_input = gr.Image(sources=["webcam"], type="numpy", label="Webcam Feed", streaming=True, width=256, height=256)
                processed_output = gr.Image(type="numpy", label="Processed Live Avatar")
                
    # Event Bindings
    source_image_input.change(
        fn=on_source_image_change,
        inputs=[source_image_input],
        outputs=[status_output]
    )
    
    calibrate_btn.click(
        fn=calibrate_camera_pose,
        inputs=[],
        outputs=[status_output]
    )
    
    apply_settings_btn.click(
        fn=change_pipeline_settings,
        inputs=[mode_radio, use_mp_checkbox, animal_checkbox],
        outputs=[status_output]
    )
    
    # Real-Time Image Stream processing
    webcam_input.stream(
        fn=process_webcam_frame,
        inputs=[
            webcam_input, 
            pasteback_checkbox, 
            stitching_checkbox, 
            relative_motion_checkbox, 
            downsample_slider,
            do_crop_checkbox,
            det_thresh_slider,
            smooth_factor_slider
        ],
        outputs=[processed_output],
        show_progress="none",
        stream_every=0.05,
        time_limit=3600
    )

if __name__ == '__main__':
    demo.queue()
    demo.launch(
        server_port=args.port,
        share=False,
        server_name=args.host_ip,
        theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")]),
        css=custom_css,
        js=js_func
    )
