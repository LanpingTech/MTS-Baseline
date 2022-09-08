from .model import base_Model, TC, NTXentLoss
from .dataset import Load_Dataset
from .config import Config

import time

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

def conv_out_size(x, k, s, p):
    return (x - k + 2 * p) // s + 1

def pool_out_size(x, k, s, p):
    return (x - k + 2 * p) // s + 1

def get_out_len(x, config:Config):
    x = conv_out_size(x, config.kernel_size, config.stride, config.kernel_size // 2)
    x = pool_out_size(x, 2, 2, 1)
    return x

def training_processing(data, config:Config, cluster_cfg, logger=None):
    x_train, y_train, x_test, y_test = data
    seq_len = x_train.shape[2]
    config.features_len = get_out_len(get_out_len(get_out_len(seq_len, config), config), config)
    config.input_channels = x_train.shape[1]

    # data processing
    trainset = Load_Dataset(x_train, config)
    loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True)

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

        features = encode(model, x_test, config)
        km = KMeans(n_clusters=cluster_cfg.n_clusters, n_init=cluster_cfg.n_init).fit(features)
        test_pred = km.labels_
        test_true = y_test
        result = cluster_cfg.metrics(test_pred, test_true)
        logger("第{}Epoch的精度为：{}".format(epoch + 1, result))

    features = encode(model, x_test, config)
    km = KMeans(n_clusters=cluster_cfg.n_clusters, n_init=cluster_cfg.n_init).fit(features)
    test_pred = km.labels_
    test_true = y_test
    result = cluster_cfg.metrics(test_pred, test_true)
    logger("最终精度为：{}".format(result))

def encode(model, X, config:Config):
    test = torch.utils.data.TensorDataset(torch.from_numpy(X).to(torch.float))
    test_generator = torch.utils.data.DataLoader(test, batch_size=config.batch_size)
    features = np.zeros((np.shape(X)[0], config.output_channels * config.features_len))
    model = model.eval()

    count = 0
    with torch.no_grad():
        for batch in test_generator:
            batch = batch[0].to(config.device)
            features[
            count * config.batch_size: (count + 1) * config.batch_size
            ] = model.predict(batch).cpu()
            count += 1
    return features

        




    


