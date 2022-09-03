from .model import base_Model, TC, NTXentLoss
from .dataset import Load_Dataset
from .config import Config

import time

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

def training_processing(data, config, encode_fn, cluster_cfg, logger=None):
    x_train, y_train, x_test, y_test = data
    config.in_channel = x_train.shape[1]

    # data processing
    trainset = Load_Dataset(x_train, config)
    loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True, drop_last=config.drop_last, num_workers=4)

    # model
    model = base_Model(config).to(config.device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=3e-4)
    temporal_contr_model = TC(config, config.device).to(config.device)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=3e-4)

    # loss function
    loss_fn = NTXentLoss(config.device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)

    # training
    for epoch in range(config.epochs):

        epoch_start=time.time()
        model = model.train()
        temporal_contr_model = temporal_contr_model.train()

        for batch_idx, (data, aug1, aug2) in enumerate(loader):
            data = data.float().to(config.device)
            aug1 = aug1.float().to(config.device)
            aug2 = aug2.float().to(config.device)

            model_optimizer.zero_grad()
            temporal_contr_optimizer.zero_grad()

            features1 = model(aug1)
            features2 = model(aug2)

            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

            zis = temp_cont_lstm_feat1 
            zjs = temp_cont_lstm_feat2

            lambda1 = 1
            lambda2 = 0.7

            nt_xent_criterion = NTXentLoss(config.device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2

            loss.backward()
            model_optimizer.step()
            temporal_contr_optimizer.step()

        epoch_end=time.time()
        logger('Epoch: {}, time: {}'.format(epoch + 1, epoch_end - epoch_start))

        features = encode_fn(model, x_test)
        km = KMeans(n_clusters=cluster_cfg.n_clusters, n_init=cluster_cfg.n_clusters, n_jobs=-1).fit(features)
        test_pred = km.labels_
        test_true = y_test
        result = cluster_cfg.metrics(test_pred, test_true)
        logger("第{}Epoch的精度为：{}".format(epoch + 1, result))

    features = encode_fn(model, x_test)
    km = KMeans(n_clusters=cluster_cfg.n_clusters, n_init=cluster_cfg.n_clusters, n_jobs=-1).fit(features)
    test_pred = km.labels_
    test_true = y_test
    result = cluster_cfg.metrics(test_pred, test_true)
    logger("最终精度为：{}".format(result))



        




    


