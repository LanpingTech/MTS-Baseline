from .model import TSTransformerEncoder, MaskedMSELoss
from .dataset import ImputationDataset, collate_unsuperv, collate_unsuperv_eval
from .config import Config

import time

import numpy as np
import torch

from sklearn.cluster import KMeans

def training_processing(data, config:Config, cluster_cfg, logger=None):
    x_train, y_train, x_test, y_test = data
    x_train = x_train.transpose(0, 2, 1)
    x_test = x_test.transpose(0, 2, 1)
    config.input_channels = x_train.shape[2]
    config.timesteps = x_train.shape[1]

    # data processing
    trainset = ImputationDataset(x_train)
    loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=4, 
                                         collate_fn=lambda x: collate_unsuperv(x, max_len=config.timesteps))

    # model
    encoder = TSTransformerEncoder(
        feat_dim=config.input_channels,
        max_len=config.timesteps,
        d_model=config.d_model, 
        n_heads=config.num_heads,
        num_layers=config.num_layers, 
        dim_feedforward=config.dim_feedforward, 
        dropout=config.dropout,
        pos_encoding=config.pos_encoding,
        activation=config.activation,
        norm=config.normalization_layer,
        freeze=config.freeze
    ).to(config.device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate, weight_decay=config.decay)

    # loss function
    loss_fn = MaskedMSELoss(reduction='none')

    # training
    for epoch in range(config.epochs):

        epoch_start=time.time()
        encoder = encoder.train()

        for batch in loader:
            X, targets, target_masks, padding_masks = batch
            targets = targets.to(config.device)
            target_masks = target_masks.to(config.device)
            padding_masks = padding_masks.to(config.device)

            optimizer.zero_grad()
            predictions = encoder(X.to(config.device), padding_masks)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = loss_fn(predictions, targets, target_masks)
            loss = loss.sum() / len(loss)

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
    test_generator = torch.utils.data.DataLoader(test, batch_size=config.batch_size, 
                                                 collate_fn=lambda x: collate_unsuperv_eval(x, max_len=config.timesteps))
    features = np.zeros((np.shape(X)[0], config.output_channels))
    model = model.eval()

    count = 0
    with torch.no_grad():
        for batch in test_generator:
            X, padding_masks = batch
            X = X.to(config.device)
            padding_masks = padding_masks.to(config.device)
            features[
            count * config.batch_size: (count + 1) * config.batch_size
            ] = model(X, padding_masks).cpu()
            count += 1
    return features

        




    


