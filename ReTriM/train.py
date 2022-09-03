from .model import Encoder, Decoder
from .dataset import TSDataset, set_triplets_batch
from .config import Config

import time

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

def get_dist(representation1, representation2):
    d=representation1.shape[0]
    size_representation=representation1.size(1)
            
    sum1 = 0
    sum2 = 0
    sum3 = 0
    
    for i in range(d):
        sum1 = sum1 + torch.bmm(representation1[i].view(1,1,size_representation), representation2[i].view(1,size_representation,1))
    
    for i in range(d):
        sum2 = sum2 + torch.pow(representation1[i], 2)
        sum3 = sum3 + torch.pow(representation2[i], 2)

    sum4 = torch.bmm((torch.pow(sum2,0.5)).view(1,1,size_representation),(torch.pow(sum3,0.5)).view(1,size_representation,1))
    dist = 1 - sum1 / sum4
    return dist

def training_processing(data, config, encode_fn, cluster_cfg, logger=None):
    x_train, y_train, x_test, y_test = data
    config.in_channels = x_train.shape[1]
    config.timesteps = x_train.shape[2]

    # data processing
    trainset = TSDataset(x_train)
    loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True, drop_last=config.drop_last, num_workers=4)

    # model
    encoder = Encoder(config.in_channels, config.timesteps, config.out_channels).to(config.device)
    encoder_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=3e-4)
    decoder = Decoder(config.in_channels, config.timesteps, config.out_channels).to(config.device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=3e-4)

    # loss function
    loss_fn = torch.nn.L1Loss(reduce=True,size_average=True)

    # training
    for epoch in range(config.epochs):

        epoch_start=time.time()
        encoder = encoder.train()
        decoder = decoder.train()

        for batch_idx, data in enumerate(loader):
            loss = 0

            Xa, Xp, Xn = set_triplets_batch(data)
            Xa = torch.from_numpy(Xa).to(config.device)
            Xp = torch.from_numpy(Xp).to(config.device)
            Xn = torch.from_numpy(Xn).to(config.device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            representation = encoder(Xa)  # Anchors representations(8,320)
            
            positive_representation = encoder(Xp)# Positive samples representations
            
            negative_representation = encoder(Xn)  # negative samples representations

            loss_Reconstruction = loss_fn(data, decoder(encoder(data)))

            pos_anchor_dist = get_dist(representation, positive_representation)
            neg_anchor_dist = get_dist(representation, negative_representation)
            
            margin = 1.0  # 调参
            scaler = torch.tensor([0]).to(config.device)
            loss_Contrastive = torch.max(scaler, margin + pos_anchor_dist - neg_anchor_dist)

            loss = 0.5 * loss_Reconstruction + 0.5 * loss_Contrastive

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        epoch_end=time.time()
        logger('Epoch: {}, time: {}'.format(epoch + 1, epoch_end - epoch_start))

        features = encode_fn(encoder, x_test)
        km = KMeans(n_clusters=cluster_cfg.n_clusters, n_init=cluster_cfg.n_clusters, n_jobs=-1).fit(features)
        test_pred = km.labels_
        test_true = y_test
        result = cluster_cfg.metrics(test_pred, test_true)
        logger("第{}Epoch的精度为：{}".format(epoch + 1, result))

    features = encode_fn(encoder, x_test)
    km = KMeans(n_clusters=cluster_cfg.n_clusters, n_init=cluster_cfg.n_clusters, n_jobs=-1).fit(features)
    test_pred = km.labels_
    test_true = y_test
    result = cluster_cfg.metrics(test_pred, test_true)
    logger("最终精度为：{}".format(result))



        




    


