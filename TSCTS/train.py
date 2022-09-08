from .model import Encoder
from .dataset import TripletSelection
from .config import Config

import time

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

def training_processing(data, config:Config, cluster_cfg, logger=None):
    x_train, y_train, x_test, y_test = data
    config.input_channels = np.shape(x_train)[1]

    # data processing
    trainset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).to(torch.float))
    loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    triplet_selection = TripletSelection(config.windows, config.theta)

    # model
    encoder = Encoder(config.input_channels, config.channels, config.depth, config.reduced_size, config.output_channels, config.kernel_size).to(config.device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=3e-4)

    # training
    for epoch in range(config.epochs):

        epoch_start=time.time()
        encoder = encoder.train()

        for batch_idx, batch in enumerate(loader):
            loss = 0
            pos_samples, neg_samples, ref_samples = triplet_selection(batch, x_train, config.nb_random_samples)

            encoder_optimizer.zero_grad()

            for i in range(len(ref_samples)):
                ref = ref_samples[i].to(config.device)
                pos_list = pos_samples[i]
                neg_list = neg_samples[i]

                ref_embedding = encoder(ref)

                for pos in pos_list:
                    pos = torch.from_numpy(pos).to(torch.float).to(config.device)
                    pos_embedding = encoder(pos)
                    loss += -torch.nn.functional.logsigmoid(torch.bmm(
                            ref_embedding.view(1, 1, config.output_channels),
                            pos_embedding.view(1, config.output_channels, 1)))

                for neg in neg_list:
                    neg = torch.from_numpy(neg).to(torch.float).to(config.device)
                    neg_embedding = encoder(neg)
                    loss += -torch.nn.functional.logsigmoid(-torch.bmm(
                            ref_embedding.view(1, 1, config.output_channels),
                            neg_embedding.view(1, config.output_channels, 1)))

            loss = loss / config.batch_size
            loss.backward()
            encoder_optimizer.step()

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



        




    


