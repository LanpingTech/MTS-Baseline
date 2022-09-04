import torch

class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 9
        self.channels = 40
        self.depth = 5
        self.reduced_size = 160
        self.output_channels = 320
        self.kernel_size = 3

        self.windows = 16
        self.theta = 0.5
        self.nb_random_samples = 8

        # training configs
        self.epochs = 30
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.learning_rate = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128