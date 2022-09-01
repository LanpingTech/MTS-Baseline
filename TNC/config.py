import torch

mc_sample_size = 40
window_size = 50
w = 0.05
augmentation = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoding_size = 320

learning_rate = 1e-3
decay=0.005
epochs = 30