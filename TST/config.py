import torch

class Config(object):
    def __init__(self):
        self.input_channels = 1
        self.timesteps = 1
        self.d_model = 64
        self.num_heads = 8
        self.num_layers = 3
        self.dim_feedforward = 256
        self.dropout = 0.1
        self.pos_encoding = 'fixed'
        self.activation = 'gelu'
        self.normalization_layer = 'BatchNorm'
        self.freeze = False

        self.mc_sample_size = 40
        self.window_size = 50
        self.w = 0.05
        self.augmentation = 5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoding_size = 320

        self.batch_size = 32
        self.learning_rate = 1e-3
        self.decay=0.005
        self.epochs = 30