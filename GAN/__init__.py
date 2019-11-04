####################
from GAN.preprocessing import DataLoader, Timer
from GAN.network import Network
from datetime import datetime
####################
import numpy as np
from numpy.random import random
from matplotlib import pyplot as plt 
import pandas as pd
import ast
import os
import cv2
####################

# GAN!!
class GenerativeAdversarialNetwork:
    '''
    https://arxiv.org/pdf/1412.6806.pdf
    -> Strides vs Pooling in Convolutional Layers
    
    https://arxiv.org/pdf/1412.6980v8.pdf 
    -> Stochastic Optimization with Adam for Generative Adversarial Network
    
    https://medium.com/@ilango100/batchnorm-fine-tune-your-booster-bef9f9493e22
    -> Batch Normalization Layer's Momentum Value

    https://github.com/vdumoulin/conv_arithmetic
    -> Convolutional Layers Visualization
    '''
    def __init__(self, option):
        self.network = Network()
        self.G = self.D = self.GAN = None
        self.option = option
        self.log = None
    
    def __enter__(self):
        if self.option == "train":
            self.G, self.D, self.GAN = self.network.build_and_compile()
            self.log = open("datafiles/output/logs/train_log_{}.txt"\
                            .format(datetime.now().strftime('%Y%m%d_%Hh%Mm')), "w+")
            return self
        
        if self.option == "waifus":
            '''
            import h5 pre-trained models. . .
            '''
            return self
        
        raise Exception("Object being constructed without a proper option.")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log.close()

        
    def noise(self, batch_size):
        noise = 0.1 * np.random.ranf((batch_size,) + (1, 1, 100))
        noise -= 0.05
        noise += np.random.randint(0, 2, size=((batch_size,) + (1, 1, 100)))
        noise -= 0.5
        noise *= 2
        return noise
        
    def train(self, epochs=5000, batch_size=64, save_interval=50):
        '''
        During the training, if D loss becomes zero/nearly zero and out performs G,
        try reducing D's learning rate and increasing D's dropout rate.
        -> .../GAN/network.py
        '''
        start = 0
        print("Training for {} epochs. . .".format(epochs))
        x = list(range(1, epochs+1)) #plot_x
        d, g = [], []
        ###
        output = plt.figure(figsize=(20, 16), dpi= 100, facecolor='w', edgecolor='k')
        iteration = 0
        ###
        for epoch in range(epochs):
            ### REAL IMAGE INPUT RANDOMIZED BATCH
            if (epoch+1) % 20 == 0 or epoch ==0:
                t = Timer() #log time taken
            data = DataLoader('/kaggle/input/waifus/waifus/images/')
            stop = start + batch_size
            face_images, err_increment = data.load_data(start) # list(not batch) of np arrays of face images
            for i in range(len(face_images)):
                face_images[i] = cv2.resize(face_images[i], (128, 128), interpolation=cv2.INTER_CUBIC)
            imageset = np.asarray(face_images).astype(np.float32) # already float32 ?? .astype(np.float32) fails
            imageset = (imageset -127.5) / 127.5 # /= 128.0 -= 1.0 or /= 255.0
            # imageset = imageset.reshape((imageset.shape[0],) + (128, 128, 3)) / 128.0 - 1.0
            real = imageset
            ###
            cv2.INTER_CUBIC
            ### NOISE IMAGE INPUT
            generated = self.G.predict(self.noise(batch_size))
            ###
            
            ### COMBINED INPUT
            # combined = np.concatenate([generated, real])
            smoothlabeledones = np.ones((batch_size, 1)) - (np.random.ranf((batch_size, ) + (1, )) * 0.1)
            smoothlabeledzeros = np.zeros((batch_size, 1)) + (np.random.ranf((batch_size, ) + (1, )) * 0.1)
            ###
            
            
            #d_loss = self.D.train_on_batch(combined, labels)
            d_loss_real = self.D.train_on_batch(real, smoothlabeledones)
            d_loss_fake = self.D.train_on_batch(generated, smoothlabeledzeros)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            
            ###
            random_latent_vectors = self.noise(batch_size)
            # 0->fake 1->real
            a_loss = self.GAN.train_on_batch(random_latent_vectors, smoothlabeledones)
            
            if (epoch+1) % 20 == 0 or epoch == 0:
                log_mesg = "{:d}: [D loss: {:f}".format(self.total_epochs+epoch+1, d_loss)
                log_mesg = "{:s}  [G loss: {:f}]".format(log_mesg, a_loss)
                print(log_mesg, end=" ")
                t.timer()
            d.append(d_loss)
            g.append(a_loss)
            
            ###TEST
            #result = model.G.predict(model.noise(1))
            #result *= 127.5 # += 1.0 *= 127.0
            #result += 127.5
            #im = result.astype(np.uint8)[0]
            #testfig.add_subplot(20, 5, iteration+1)
            #plt.imshow(im)
            #iteration += 1
            ###
            
            ### ITERATING BATCHES THROUGH DATASET
            start += batch_size
            start += err_increment
            if start > len(imageset) - batch_size:
                start = 0
            if epoch != 0 and epoch % save_interval == 0:
                result = self.G.predict(self.noise(1))
                result *= 127.5
                result += 127.5
                sample_image = result.astype(np.uint8)[0]
                output.add_subplot(int((epochs/save_interval)/5)+1, 5, iteration+1)
                plt.imshow(sample_image)
                iteration += 1
        self.train_iterations += 1
        self.train_epochs.append(epochs)
        plt.show()
        return x, d, g
    
    def generate_images(self, batch_size=20):
        if self.train_iterations < 1:
            print("Please train the model at least once for the model to generate an image.")
        else:
            fig_x = plt.figure(figsize=(8, 12), dpi= 120, facecolor='w', edgecolor='k')
            result = self.G.predict(self.noise(batch_size))
            result *= 127.5 # += 1.0 *= 127.0 or &= 255.0
            result += 127.5
            im = result.astype(np.uint8)
            iteration = 1
            for image in im:
                fig_x.add_subplot(5, 4, iteration)
                iteration += 1
                plt.imshow(image)
            plt.show()
        