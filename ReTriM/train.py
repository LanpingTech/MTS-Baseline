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

def training_processing(data, config:Config, cluster_cfg, logger=None):
    x_train, y_train, x_test, y_test = data
    config.in_channels = x_train.shape[1]
    config.timesteps = x_train.shape[2]

    # data processing
    trainset = TSDataset(x_train)
    loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # model
    encoder = Encoder(config.in_channels, config.timesteps, config.output_channels).to(config.device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=3e-4)
    decoder = Decoder(config.in_channels, config.timesteps, config.output_channels).to(config.device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=3e-4)

    # loss function
    loss_fn = torch.nn.L1Loss(reduction='mean')
    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    # training
    for epoch in range(config.epochs):

        epoch_start=time.time()
        encoder = encoder.train()
        decoder = decoder.train()

        for batch_idx, data in enumerate(loader):
            loss = 0
            
            Xa, Xp, Xn = set_triplets_batch(data)
            data = data.to(torch.float).to(config.device)
            Xa = torch.from_numpy(Xa).to(torch.float).to(config.device)
            Xp = torch.from_numpy(Xp).to(torch.float).to(config.device)
            Xn = torch.from_numpy(Xn).to(torch.float).to(config.device)
           
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            representation = encoder(Xa)  # Anchors representations(batch_size,output_channels)
            
            positive_representation = encoder(Xp)# Positive samples representations
            
            negative_representation = encoder(Xn)  # negative samples representations
            
            loss_Reconstruction = loss_fn(data, decoder(encoder(data)))
            
            triplet_loss= torch.nn.TripletMarginWithDistanceLoss(distance_function=get_dist, margin=0.8)
            loss_Contrastive = triplet_loss(representation, positive_representation, negative_representation)           

            alpha = 0.5
            loss = alpha * loss_Reconstruction + (1-alpha) * loss_Contrastive
           
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
        
        epoch_end=time.time()
        logger('Epoch: {}, time: {}'.format(epoch + 1, epoch_end - epoch_start))

        features = encode(encoder, x_test, config)
        km = KMeans(n_clusters=cluster_cfg.n_clusters, n_init=cluster_cfg.n_clusters).fit(features)
        test_pred = km.labels_
        test_true = y_test
        result = cluster_cfg.metrics(test_pred, test_true)
        logger("第{}Epoch的精度为：{}".format(epoch + 1, result))

    features = encode(encoder, x_test, config)
    km = KMeans(n_clusters=cluster_cfg.n_clusters, n_init=cluster_cfg.n_clusters).fit(features)
    test_pred = km.labels_
    test_true = y_test
    result = cluster_cfg.metrics(test_pred, test_true)
    logger("最终精度为：{}".format(result))

def encode(model, X, config:Config):
    test = torch.utils.data.TensorDataset(torch.from_numpy(X).to(torch.float))
    test_generator = torch.utils.data.DataLoader(test, batch_size=config.batch_size)
    features = np.zeros((np.shape(X)[0], config.output_channels))
    model = model.eval()

    count = 0
    with torch.no_grad():
        for batch in test_generator:
            batch = batch[0].to(config.device)
            features[
            count * config.batch_size: (count + 1) * config.batch_size
            ] = model(batch).cpu()
            count += 1
    return features

        




    


