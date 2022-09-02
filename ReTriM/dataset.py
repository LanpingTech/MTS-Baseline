import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import pairwise_distances

def set_triplets_batch(batch):
    batch_size, input_dim, time_steps = batch.shape
    Xa = np.empty((0, input_dim, time_steps))
    Xp = np.empty((0, input_dim, time_steps))
    Xn = np.empty((0, input_dim, time_steps))
    Xb=batch.reshape(batch_size,-1)
    dist = pairwise_distances(Xb)
    for i in range(batch_size):
        j = dist[i, :].argmax()  # 最大索引
        k = np.argsort(dist[i, :])[1]  # 第二小索引
        Xa= np.concatenate((Xa, batch[i:i + 1]), axis=0)
        Xp= np.concatenate((Xp, batch[j:j + 1]), axis=0)
        Xn= np.concatenate((Xn, batch[k:k + 1]), axis=0)
        
    return Xa, Xp, Xn
   
class TSDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return np.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index]

