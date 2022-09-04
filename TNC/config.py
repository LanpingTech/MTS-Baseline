import torch

class Config(object):
    def __init__(self):
        self.mc_sample_size = 40
        self.window_size = 50
        self.w = 0.05
        self.augmentation = 5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoding_size = 320

        self.learning_rate = 1e-3
        self.decay=0.005
        self.epochs = 30