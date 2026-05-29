import os
import urllib.request
import urllib.error

def download(url, path):
    print(f"Downloading {url} to {path}...")
    try:
        urllib.request.urlretrieve(url, path)
        print("Done.")
    except urllib.error.URLError as e:
        print(f"Failed to download: {e}")

os.makedirs('checkpoints/liveportrait_pytorch', exist_ok=True)
download('https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/base_models/warping_module.pth', 'checkpoints/liveportrait_pytorch/warping_module.pth')
download('https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/base_models/spade_generator.pth', 'checkpoints/liveportrait_pytorch/spade_generator.pth')
