from .model import TS2Vec
from .config import Config

import time

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

def training_processing(data, config, cluster_cfg, logger=None):
    x_train, y_train, x_test, y_test = data
    x_train = x_train.transpose(0, 2, 1)
    x_test = x_test.transpose(0, 2, 1)
    config.in_channels = x_train.shape[2]
    config.timesteps = x_train.shape[1]

    encoder = TS2Vec(input_dims=config.in_channels, 
                     output_dims=config.out_channels, 
                     device=config.device,
                     lr=config.learning_rate,
                     batch_size=config.batch_size,)
    
    loss_log = encoder.fit(train_data=x_train,
                           n_epochs=config.epochs,
                           verbose=True,
                           logger=logger,
                           val_data=x_test,
                           val_label=y_test,
                           cluster_cfg=cluster_cfg)

    features = encoder.encode(x_test, encoding_window='full_series')
    km = KMeans(n_clusters=cluster_cfg.n_clusters, n_init=cluster_cfg.n_clusters).fit(features)
    test_pred = km.labels_
    test_true = y_test
    result = cluster_cfg.metrics(test_pred, test_true)
    logger("最终精度为：{}".format(result))



        




    


