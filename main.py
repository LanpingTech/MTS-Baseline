import os
from select import select
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import importlib
import json

import random
import numpy
import torch

from metrics import ClusterConfig

def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_data(data_name):
    path = '../../Data/new_data/{}/'.format(data_name)
    x_train = numpy.load(path + 'X_train.npy').astype(numpy.float32)
    y_train = numpy.load(path + 'y_train.npy')
    x_test = numpy.load(path + 'X_test.npy').astype(numpy.float32)
    y_test = numpy.load(path + 'y_test.npy')

    return x_train, y_train, x_test, y_test

def get_logger(model_name, data_name):
    def logger(logstr):
        result_file_open = open(model_name + '/log/' + data_name + '.log', 'a')
        result_file_open.write(logstr+'\n')
        print(logstr)
        result_file_open.close()
    return logger

if __name__ == '__main__':

    seed_torch(42)

    data_names = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'ERing','EthanolConcentration', 'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery', 'NATOPS', 'PEMS-SF', 'PenDigits', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']

    TST = ['EigenWorms'] # o

    TNC = ['DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'ERing','EthanolConcentration', 'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery', 'NATOPS', 'PEMS-SF', 'PenDigits', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']#gpu1

    TSTCC = ['InsectWingbeat', 'JapaneseVowels', 'PenDigits', 'PhonemeSpectra', 'RacketSports'] # o

    ReTriM = ['EigenWorms', 'Epilepsy', 'EthanolConcentration', 'FaceDetection', 'HandMovementDirection', 'Heartbeat', 'InsectWingbeat', 'Libras', 'NATOPS', 'RacketSports', 'SelfRegulationSCP2', 'StandWalkJump'] #  o
    
    TestData=['Epilepsy']
    
    TS2Vec = ['Epilepsy', 'ERing','EthanolConcentration', 'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery', 'NATOPS', 'PEMS-SF', 'PenDigits', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary'] # o 'EigenWorms', 

    TSCTS = ['AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'ERing','EthanolConcentration', 'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery', 'NATOPS', 'PEMS-SF', 'PenDigits', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary'] # gpu3

    model_name = 'ReTriM'

    for data_name in TestData:

        params = json.load(open('hyper/' + data_name + '/' + data_name + '_hyperparameters.json', 'r'))

        config = getattr(importlib.import_module(model_name + '.config'), 'Config')()
        config.batch_size = params['batch_size']
        config.epochs = params['epochs']

        config.depth = params['depth']

        cluster_cfg = ClusterConfig(params['n_clusters'], params['n_init'])

        data = load_data(data_name)
        logger = get_logger(model_name, data_name)

        logger(data_name)
        try:
            training_func = getattr(importlib.import_module(model_name + '.train'), 'training_processing')
            training_func(data, config, cluster_cfg, logger)
        except Exception as e:
            continue
        logger('=' * 50)
        logger(' ')
        logger(' ')
        torch.cuda.empty_cache()

# nohup python -u main.py > gpu0.log 2>&1 &