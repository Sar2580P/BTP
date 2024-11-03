import torch.nn as nn
import torch
import os
import glob

class SDA_Initialiser(nn.Module):
    def __init__(self, config):
        super(SDA_Initialiser, self).__init__()
        # [1280, 800, 400, 100]
        self.config = config['stage_1']['SDA_params']
        self.base_model, self.encoder, self.decoder = self.get_model()
        self.save_ckpt_path = None
        self.activation = nn.PReLU() 
        
    def check_file_exists(self, file_pattern):
        return len(glob.glob(file_pattern))>0
        
    def get_model(self):
        layers = nn.ModuleList()
        ckpt_pattern = f"{self.config["save_dir"]}/stage_1/SDA_init_{self.config['input_dim']}"
        last_idx = -1 ; 
        for i, hidden_dims in enumerate(self.config['hidden_dims']):
            if not self.check_file_exists(f"{ckpt_pattern}_{hidden_dims}*"):
                break
            ckpt_pattern += f"_{hidden_dims}"
            last_idx = i
            layers.append(nn.Linear(self.config['hidden_dims'][i-1], self.config['hidden_dims'][i]))
        ckpt_pattern += ".ckpt"
        
        base_model = None
        if os.path.exists(ckpt_pattern):     # source of bug, need to write mapping for each dict state
            base_model = nn.Sequential(*layers)
            base_model.load_state_dict(torch.load(ckpt_pattern))
            base_model.eval()
            
        encoder, decoder = None, None
        if last_idx==0:
            encoder = nn.Linear(self.config['input_dim'], self.config['hidden_dims'][last_idx])
            decoder = nn.Linear(self.config['hidden_dims'][last_idx], self.config['input_dim'])
        
        if encoder is not None : 
            self.save_ckpt_path = ckpt_pattern[:-5]+f"_{self.config['hidden_dims'][last_idx+1]}.ckpt"
        return base_model, encoder, decoder
    
    def forward(self, x: torch.Tensor):
        if self.base_model is not None:
            x = self.base_model(x)
        if self.encoder is not None:
            # add noise by masking some of the input features to 0 randomly
            mask:torch.Tensor = torch.rand(x.shape, device=x.device) > self.config['mask_fraction']
            x_masked = x * mask.float()
            y = self.encoder(x_masked)
            y = self.activation(y)
            y = self.decoder(y)
            return y
        return x
    