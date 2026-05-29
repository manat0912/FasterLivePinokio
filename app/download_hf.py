import os
from huggingface_hub import hf_hub_download
import shutil

os.makedirs('checkpoints/liveportrait_pytorch', exist_ok=True)
print("Downloading warping_module.pth...")
w_path = hf_hub_download(repo_id='KlingTeam/LivePortrait', filename='liveportrait/base_models/warping_module.pth')
shutil.copy(w_path, 'checkpoints/liveportrait_pytorch/warping_module.pth')

print("Downloading spade_generator.pth...")
s_path = hf_hub_download(repo_id='KlingTeam/LivePortrait', filename='liveportrait/base_models/spade_generator.pth')
shutil.copy(s_path, 'checkpoints/liveportrait_pytorch/spade_generator.pth')
print("Done.")
