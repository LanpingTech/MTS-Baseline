import torch

class Config(object):
    def __init__(self):
        # model configs
        self.output_channels = 10

        # training configs
        self.epochs = 30
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.learning_rate = 1e-3

        self.batch_size = 128