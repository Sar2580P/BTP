import torch.nn as nn
import torch
import os
import glob

class SDA_Initialiser(nn.Module):
    def __init__(self, config):
        super(SDA_Initialiser, self).__init__()

        self.config = config
        self.name = "SDA_init"
        self.ckpt_file_name = None
        self.base_model, self.encoder, self.decoder = self.get_model()
        self.activation = nn.PReLU()

        self.register_buffer('mask_fraction', torch.tensor(self.config['SDA_params']['mask_fraction']))

    def check_file_exists(self, file_pattern):
        return len(glob.glob(file_pattern))>0

    def get_model(self):
        layers = nn.ModuleList()
        ckpt_dir = f"{self.config['save_dir']}/ckpts"
        ckpt_pattern = f"{self.name}_{self.config['SDA_params']['input_dim']}"
        last_idx = -1
        for i, hidden_dims in enumerate(self.config['SDA_params']['hidden_dims']):   # [800, 400, 100]
            print(ckpt_pattern, f"{ckpt_dir}/{ckpt_pattern}_{hidden_dims}*")
            if not self.check_file_exists(os.path.join(ckpt_dir, f"{ckpt_pattern}_{hidden_dims}*")):
                break
            ckpt_pattern += f"_{hidden_dims}"
            last_idx = i
            in_dim = self.config['SDA_params']['hidden_dims'][i-1] if i>0 else self.config['SDA_params']['input_dim']
            layers.append(nn.Linear(in_dim, self.config['SDA_params']['hidden_dims'][i]))

        base_model = None
        ckpt_path = glob.glob(os.path.join(ckpt_dir, ckpt_pattern+"*.ckpt"))
        if len(ckpt_path)>0:     # source of bug, need to write mapping for each dict state
            base_model = self.load_base_model(layers , ckpt_path[0])
            base_model.eval()

        encoder, decoder = None, None
        if last_idx<len(self.config['SDA_params']['hidden_dims'])-1:
            in_dim = self.config['SDA_params']['input_dim'] if last_idx==-1 else self.config['SDA_params']['hidden_dims'][last_idx]
            encoder = nn.Linear(in_dim, self.config['SDA_params']['hidden_dims'][last_idx+1])
            decoder = nn.Linear(self.config['SDA_params']['hidden_dims'][last_idx+1], in_dim)

        if encoder is not None :
            self.ckpt_file_name = ckpt_pattern+f"_{self.config['SDA_params']['hidden_dims'][last_idx+1]}"
        return base_model, encoder, decoder

    def load_base_model(self, layers, ckpt):
        state = torch.load(ckpt)

        prev_encoder_weights = {k.replace('model.encoder.', ''): v for k, v in state['state_dict'].items() if 'model.encoder.' in k}
        layers[-1].load_state_dict(prev_encoder_weights)
        for i, layer in enumerate(layers[:-1]):
            base_model_weights = {k.replace(f'model.base_model.{i}.', ''): v for k, v in state['state_dict'].items() \
                                  if f'model.base_model.{i}' in k}
            layer.load_state_dict(base_model_weights)
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        y = x
        if self.base_model is not None:
            y = self.base_model(x)
        if self.encoder is not None:
            # add noise by masking some of the input features to 0 randomly
            mask:torch.Tensor = torch.rand(y.shape, device=y.device) > self.mask_fraction
            y_masked = y * mask.to(y.dtype)
            z = self.encoder(y_masked)
            z = self.activation(z)
            y_hat = self.decoder(z)
            return y , y_hat
        return y, y
    
class Stage_Classifier(nn.Module):
    def __init__(self, config):
        super(Stage_Classifier, self).__init__()

        self.config = config
        self.name = "Stage_classifier"
        self.model = self.get_model()
        
    def get_model(self):
        layers = nn.ModuleList()
        ckpt_pattern = f"SDA_init_{self.config['SDA_params']['input_dim']}"
        for i, hidden_dims in enumerate(self.config['Stage_classifier']['hidden_dims']):
            if i==0:
                layers.append(nn.Linear(self.config['Stage_classifier']['input_dim'], hidden_dims))
            else:
                layers.append(nn.Linear(self.config['Stage_classifier']['hidden_dims'][i-1], hidden_dims))
            ckpt_pattern += f"_{hidden_dims}"
        layers.append(nn.Linear(self.config['Stage_classifier']['hidden_dims'][-1], self.config['Stage_classifier']['num_classes']))

        ckpt_path = glob.glob(os.path.join(f"{self.config['save_dir']}/ckpt", f"ckpts/{ckpt_pattern}*.ckpt"))
        assert len(ckpt_path)>0, f"Checkpoint file not found for {ckpt_pattern}"
        self.initialise_model(layers, ckpt_path[0])
        
        # Now add additional layers for robustness
        full_layers = nn.ModuleList()
        for i, layer in enumerate(layers[:-1]):  # Skip final layer initially
            full_layers.append(layer)
            
            # Add BatchNorm, Activation, and Dropout (if specified) after each linear layer
            full_layers.append(nn.BatchNorm1d(hidden_dims[i]))   # Batch normalization
            full_layers.append(nn.ReLU())                        # Activation layer
            if 'dropout' in self.config['Stage_classifier'] and self.config['Stage_classifier']['dropout'] > 0:
                full_layers.append(nn.Dropout(self.config['Stage_classifier']['dropout']))  # Dropout if specified
        
        # Finally, add the output layer
        full_layers.append(layers[-1])
        return nn.Sequential(*full_layers)
        
        
    
    def initialise_model(self, layers, ckpt):
        state = torch.load(ckpt)
        prev_encoder_weights = {k.replace('model.encoder.', ''): v for k, v in state['state_dict'].items() if 'model.encoder.' in k}
        layers[-1].load_state_dict(prev_encoder_weights)
        for i, layer in enumerate(layers[:-1]):
            base_model_weights = {k.replace(f'model.base_model.{i}.', ''): v for k, v in state['state_dict'].items() \
                                  if f'model.base_model.{i}' in k}
            layer.load_state_dict(base_model_weights)
        return 
    
    def forward(self, x: torch.Tensor):
        return self.model(x)


if __name__=="__main__":
    ckpt ="results/ckpts/SDA_init_1280_800_epoch=19-val_SDA_squared_error=152.79.ckpt"
    state = torch.load(ckpt)
    print(state['state_dict'].keys())