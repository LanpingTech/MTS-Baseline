from TNC.train import training_processing

from TNC import config

config.encoding_size = 1
print(config.__dict__)
training_processing(1)