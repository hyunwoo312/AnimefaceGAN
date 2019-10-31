####################
from GAN.preprocessing import DataLoader, AnimeFace, Timer
from GAN.Network import Network
from datetime import datetime
####################
import numpy as np
from numpy.random import random
import pandas as pd
import matplotlib.pyplot as plt
import ast # datetime
import os # file system

import matplotlib.pyplot as plt
print('File Tree', end=': ')
for dirname, _, filenames in os.walk('/GAN'):
    print(dirname)

class GenerativeAdversarialNetwork:
    '''
    https://arxiv.org/pdf/1412.6806.pdf
    -> Strides vs Pooling in Convolutional Layers
    
    https://arxiv.org/pdf/1412.6980v8.pdf 
    -> Stochastic Optimization with Adam for Generative Adversarial Network
    
    https://medium.com/@ilango100/batchnorm-fine-tune-your-booster-bef9f9493e22
    -> Batch Normalization Layer's Momentum Value
    '''
    def __init__(self, option):
        self.network = Network()
        self.G = self.D = self.GAN = None
        self.option = option
        self.log = None
    
    def __enter__(self):
        if self.option == "train":
            self.G, self.D, self.GAN = self.network.build_and_compile()
            self.log = open("datafiles/output/train_log{}.txt"\
                            .format(datetime.now().strftime('%Y%m%d_%Hh%Mm')), "w+")
            return self.log
        
        if self.option == "waifus":
            '''
            import h5 pre-trained models. . .
            '''
            return self.G
        
        raise Exception("Object being constructed without a proper option.")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log.close()
        