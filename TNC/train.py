from .model import RnnEncoder, Discriminator
from .dataset import TNCDataset
from .config import Config

import time

import numpy as np
import torch

from sklearn.cluster import KMeans

def training_processing(data, config:Config, cluster_cfg, logger=None):
    x_train, y_train, x_test, y_test = data
    in_channel = x_train.shape[1]

    # data processing
    trainset = TNCDataset(x=torch.Tensor(x_train), mc_sample_size=config.mc_sample_size, window_size=config.window_size, augmentation=config.augmentation, adf=True)
    loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=3)

    # model
    encoder = RnnEncoder(hidden_size=100, in_channel=in_channel, encoding_size=config.encoding_size, device=config.device).to(config.device)
    disc_model = Discriminator(input_size=config.encoding_size, device=config.device).to(config.device)
    params = list(disc_model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.decay)

    # loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # training
    for epoch in range(config.epochs):

        epoch_start=time.time()
        encoder = encoder.train()
        disc_model = disc_model.train()

        for x_t, x_p, x_n, _ in loader:
            mc_sample = x_p.shape[1]
            batch_size, f_size, len_size = x_t.shape
            x_p = x_p.reshape((-1, f_size, len_size))
            x_n = x_n.reshape((-1, f_size, len_size))
            x_t = np.repeat(x_t, mc_sample, axis=0)
            neighbors = torch.ones((len(x_p))).to(config.device)
            non_neighbors = torch.zeros((len(x_n))).to(config.device)
            x_t, x_p, x_n = x_t.to(config.device), x_p.to(config.device), x_n.to(config.device)

            z_t = encoder(x_t)
            z_p = encoder(x_p)
            z_n = encoder(x_n)

            d_p = disc_model(z_t, z_p)
            d_n = disc_model(z_t, z_n)

            p_loss = loss_fn(d_p, neighbors)
            n_loss = loss_fn(d_n, non_neighbors)
            n_loss_u = loss_fn(d_n, neighbors)
            loss = (p_loss + config.w * n_loss_u + (1 - config.w) * n_loss)/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_end=time.time()
        logger('Epoch: {}, time: {}'.format(epoch + 1, epoch_end - epoch_start))

        features = encode(encoder, x_test, config)
        km = KMeans(n_clusters=cluster_cfg.n_clusters, n_init=cluster_cfg.n_init).fit(features)
        test_pred = km.labels_
        test_true = y_test
        result = cluster_cfg.metrics(test_pred, test_true)
        logger("第{}Epoch的精度为：{}".format(epoch + 1, result))

    features = encode(encoder, x_test, config)
    km = KMeans(n_clusters=cluster_cfg.n_clusters, n_init=cluster_cfg.n_init).fit(features)
    test_pred = km.labels_
    test_true = y_test
    result = cluster_cfg.metrics(test_pred, test_true)
    logger("最终精度为：{}".format(result))

def encode(model, X, config:Config):
    test = torch.utils.data.TensorDataset(torch.from_numpy(X).to(torch.float))
    test_generator = torch.utils.data.DataLoader(test, batch_size=config.batch_size)
    features = np.zeros((np.shape(X)[0], config.encoding_size))
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

        




    


