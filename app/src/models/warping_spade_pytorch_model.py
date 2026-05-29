import os
import yaml
import torch
import numpy as np

from .base_model import BaseModel
from .pytorch_modules.warping_network import WarpingNetwork
from .pytorch_modules.spade_generator import SPADEDecoder
from .predictor import numpy_to_torch_dtype_dict

class WarpingSpadePyTorchModel(BaseModel):
    """
    WarpingSpade PyTorch Wrapper Model
    Bypasses TensorRT/ONNX Runtime to execute the exact original architecture
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.predict_type = kwargs.get("predict_type", "pt")
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.predictor = None # Bypass BaseModel predictor
        self.cudaStream = torch.cuda.current_stream().cuda_stream if torch.cuda.is_available() else None
        
        # Load the models config
        config_path = os.path.join(os.path.dirname(__file__), '../../LivePortrait_original/src/config/models.yaml')
        with open(config_path, 'r') as f:
            models_config = yaml.safe_load(f)['model_params']
            
        warping_params = models_config['warping_module_params']
        spade_params = models_config['spade_generator_params']

        # Instantiate original PyTorch architectures
        self.W = WarpingNetwork(**warping_params).to(self.device).eval()
        self.G = SPADEDecoder(**spade_params).to(self.device).eval()

        # Load weights
        ckpt_W_path = os.path.join('checkpoints', 'liveportrait_pytorch', 'warping_module.pth')
        ckpt_G_path = os.path.join('checkpoints', 'liveportrait_pytorch', 'spade_generator.pth')
        
        self.W.load_state_dict(torch.load(ckpt_W_path, map_location=self.device, weights_only=True))
        self.G.load_state_dict(torch.load(ckpt_G_path, map_location=self.device, weights_only=True))

    def input_process(self, *data):
        feature_3d, kp_source, kp_driving = data
        return feature_3d, kp_driving, kp_source

    def output_process(self, out):
        # Move Bx3xHxW image to BxHxWx3 numpy/torch depending on downstream needs
        out = out.permute(0, 2, 3, 1)
        out = torch.clip(out, 0, 1) * 255
        return out[0]

    def predict(self, *data):
        # data input comes from FasterLivePortrait pipeline which might be numpy or torch
        data = self.input_process(*data)
        
        torch_data = []
        for d in data:
            if isinstance(d, np.ndarray):
                torch_data.append(torch.from_numpy(d).to(self.device).float())
            else:
                torch_data.append(d.to(self.device).float())
        
        feature_3d, kp_driving, kp_source = torch_data
        
        with torch.no_grad():
            ret_dct = self.W(feature_3d, kp_driving, kp_source)
            out = self.G(ret_dct['out'])
            
        # The output process of FasterLivePortrait expects output from W+G combo
        outputs = self.output_process(out)
        return outputs
