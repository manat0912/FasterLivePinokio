import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.join(os.getcwd(), 'app'))
sys.path.append(os.path.join(os.getcwd(), 'app', 'src'))

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.config.inference_config import InferenceConfig

cfg = InferenceConfig()
pipeline = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)

# Load source image (Obama)
src_img = cv2.imread(r'C:\Users\manat\.gemini\antigravity\brain\7255ef5f-753d-4018-980f-8491a79851c2\.tempmediaStorage\media_7255ef5f-753d-4018-980f-8491a79851c2_1779930858075.mp4')
# Wait, the source image was an mp4? No, the source image is usually Obama.
# Let's just use any frame as source and another as driving.
src_img = cv2.imread('video_frames/f_0000.jpg')
ret = pipeline.prepare_source(src_img)

for i in range(55, 65):
    img = cv2.imread(f'video_frames/f_{i:04d}.jpg')
    img_crop, out_crop, out_org, dri_motion_info = pipeline.run(
        img, 
        pipeline.src_imgs[0], 
        pipeline.src_infos[0],
        first_frame=(i==55)
    )
    pitch = dri_motion_info[0]['pitch']
    yaw = dri_motion_info[0]['yaw']
    roll = dri_motion_info[0]['roll']
    std = np.std(out_crop)
    print(f"Frame {i}: pitch={pitch[0][0]:.3f}, yaw={yaw[0][0]:.3f}, roll={roll[0][0]:.3f}, std={std:.3f}")
