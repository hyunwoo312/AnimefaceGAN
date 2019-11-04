import warnings
warnings.filterwarnings("ignore")

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Conv2D, Conv2DTranspose
#from keras.layers import UpSampling2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import LeakyReLU, Dropout, GaussianNoise
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.initializers import RandomNormal
import sys

class Network:
    def __init__(self):
        self.Discriminator = None
        self.Generator = None
        self.AdversarialNetwork = None
        self.compiled = False
        self.imagesize = (128, 128, 3) # (64, 64, 3)
        self.noiseshape = (1, 1, 100)
        self.dropoutrate_d = 0.35 ## 0.3
        self.leakyreluparam = 0.2
        self.batchnormmomentum= 0.8 ## 0.8 
        self.kernel_initializer = RandomNormal(mean=0.0, stddev=0.02) #glorot_uniform is default
        self.adam_d = Adam(lr=0.0002, beta_1=0.5) #0.002 or 0.0002?
        self.adam_g = Adam(lr=0.0002, beta_1=0.5) #different learning rate for gan?
        self.loss = "binary_crossentropy"

    
    def build_discriminator(self) -> None:
        if self.Discriminator:
            return
        input_tensor = Input(shape=self.imagesize)
        '''
        ^^^ Input Layer Definition
        vvv Output Layer Definition
        '''
        ### Simple layering definition for repititive layer stacking
        def repeatlayering(layers, filters, shape=(4, 4), batch=True):
            x = Conv2D(
                    filters, 
                    shape, 
                    strides=2, 
                    padding="same", 
                    kernel_initializer=self.kernel_initializer) (layers)
            if batch:
                x = BatchNormalization(momentum=self.batchnormmomentum) (x)
            x = LeakyReLU(alpha=self.leakyreluparam) (x)
            x = Dropout(self.dropoutrate_d) (x)
            return x
        
        ### Layering
        l = input_tensor
        #128x128
        l = GaussianNoise(0.1) (l)
        l = repeatlayering(l, 64, batch=False)
        #64x64
        l = repeatlayering(l, 128)
        #32x32
        l = repeatlayering(l, 256)
        #16x16
        l = repeatlayering(l, 512)
        #8x8
        l = repeatlayering(l, 1024)
        #4x4
        l = Flatten() (l)
        ### sigmoid activation output layer for discriminator NN
        l = Dense(1, kernel_initializer=self.kernel_initializer) (l)
        l = Activation("sigmoid") (l) #hard_sigmoid?
        self.Discriminator = Model(inputs=input_tensor, outputs=l)
        self.Discriminator.name = "Discriminator"
    
    
    def build_generator(self) -> None:
        if self.Generator:
            return
        input_tensor = Input(shape=self.noiseshape)
        '''
        ^^^ Input Layer Definition
        VVV Output Layer Definition
        '''
        ### Simple layering definition for repititive layer stacking
        def repeatlayering(layers, filters, shape=(4, 4), paddingval='same', strides=True):
            if strides:
                x = Conv2DTranspose(
                        filters, 
                        shape, 
                        padding=paddingval, 
                        strides=(2, 2), 
                        kernel_initializer=self.kernel_initializer) (layers)
                #x = UpSampling2D((2, 2)) (layers)
                #x = Conv2D(filters, shape, padding=paddingval) (x)
            else:
                x = Conv2DTranspose(
                        filters, 
                        shape, 
                        padding=paddingval, 
                        kernel_initializer=self.kernel_initializer) (layers)
            x = BatchNormalization(momentum=self.batchnormmomentum) (x)
            x = LeakyReLU(alpha=self.leakyreluparam) (x)
            return x
        
        ### Layering
        l = input_tensor
        l = repeatlayering(l, 1024, paddingval="valid", strides=False)
        l = repeatlayering(l, 512)
        l = repeatlayering(l, 256) 
        l = repeatlayering(l, 128)
        l = repeatlayering(l, 64)
        l = Conv2D(64, (3, 3), padding='same', kernel_initializer=self.kernel_initializer) (l)
        l = BatchNormalization(momentum=self.batchnormmomentum) (l)
        l = LeakyReLU(alpha=self.leakyreluparam) (l)
        l = Conv2DTranspose(3, (4, 4), padding='same', strides=(2, 2), kernel_initializer=self.kernel_initializer) (l)
        l = Activation("tanh") (l) #tanh
        # output should be 128x128 at this point
        self.Generator = Model(inputs=input_tensor, outputs=l)
        self.Generator.name = "Generator"
    
    
    def build_and_compile(self):
        if self.compiled:
            return self.Generator, self.Discriminator, self.AdversarialNetwork
        if not self.Discriminator:
            self.build_discriminator()
        # Compile. . .
        self.Discriminator.compile(self.adam_d, loss=self.loss)
        self.Discriminator.trainable = False
        # Discriminator won't be trained during GAN training
        if not self.Generator:
            self.build_generator()
        noise = Input(shape=self.noiseshape)
        gen = self.Generator(noise)
        output = self.Discriminator(gen)
        self.AdversarialNetwork = Model(inputs=noise, outputs=output)
        self.AdversarialNetwork.name = "Generative Adversarial Network"
        self.AdversarialNetwork.compile(self.adam_g, loss=self.loss)
        ### redirecting keras model architecture printing to a file
        self.compiled = True
        original_stdout = sys.stdout
        f = open("datafiles/models/current_GAN_architecture.txt", "w+")
        sys.stdout = f
        self.Generator.summary()
        self.Discriminator.summary()
        self.AdversarialNetwork.summary()
        sys.stdout = original_stdout
        f.close()
        ###
        print(
'''
The current architecture for the GAN has been saved to
.../GAN/datafiles/models/current_GAN_architecture.txt
Caffe prototext type format will be supported later(?)
'''
        )
        return self.Generator, self.Discriminator, self.AdversarialNetwork

n = Network()
n.build_and_compile()