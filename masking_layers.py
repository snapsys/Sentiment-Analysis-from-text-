from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv1D, Dense, Flatten, Activation
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.optimizers import Adam, Adagrad, SGD
from keras import backend as K
import scipy.io as sio
import numpy as np
import scipy.sparse as sparse
from keras import regularizers
from keras.constraints import maxnorm, non_neg
import keras
import sys
import time
import h5py, pickle
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import metrics
from keras.layers.pooling import _GlobalPooling1D
from keras.engine.topology import Layer, InputSpec

class GlobalMaskedAveragePooling1D(Layer):

    def __init__(self, **kwargs):
        super(GlobalMaskedAveragePooling1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        print("input shapes to masking layer is {}".format(input_shape))
        return (input_shape[0][0], input_shape[0][2])
    

    def call(self, inputs):
        x, mask = inputs
        N = K.sum(mask, axis=1, keepdims=False)
        #print(x.shape, mask.shape, N)

        return  K.sum(x*mask, axis=1, keepdims=False)/N


class GlobalMaskedAveragePooling1D_Conv1(Layer):

    def __init__(self, **kwargs):
        super(GlobalMaskedAveragePooling1D_Conv1, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        print("input shapes to masking layer is {}".format(input_shape))
        return (input_shape[0][0], input_shape[0][2])
    

    def call(self, inputs):
        x, mask = inputs
        N = K.sum(mask, axis=1, keepdims=False)
        #print(x.shape, mask.shape, N)
        mask_1 = np.ones((100,1), dtype=int)
        mask_1[::2] = 0
        mask_2 = 1 - mask_1
        mask_1 = K.variable(mask_1)
        mask_2 = K.variable(mask_2)
        return  K.sum(x*mask*mask_1, axis=1, keepdims=False)/N

class GlobalMaskedAveragePooling1D_Conv2(Layer):

    def __init__(self, **kwargs):
        super(GlobalMaskedAveragePooling1D_Conv2, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        print("input shapes to masking layer is {}".format(input_shape))
        return (input_shape[0][0], input_shape[0][2])
    

    def call(self, inputs):
        x, mask = inputs
        N = K.sum(mask, axis=1, keepdims=False)
        #print(x.shape, mask.shape, N)
        mask_1 = np.ones((100,1), dtype=int)
        mask_1[::2] = 0
        mask_2 = 1 - mask_1
        mask_1 = K.variable(mask_1)
        mask_2 = K.variable(mask_2)
        return  K.sum(x*mask*mask_2, axis=1, keepdims=False)/N


class GlobalMaskedAveragePooling1D_Spkr1(Layer):

    def __init__(self, **kwargs):
        super(GlobalMaskedAveragePooling1D_Spkr1, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        print("input shapes to masking layer is {}".format(input_shape))
        return (input_shape[0][0], input_shape[0][2])
    

    def call(self, inputs):
        x, SpkrID = inputs
        mask = K.equal(K.sum(K.abs(SpkrID), axis=2, keepdims=False), 1)  
        mask = K.cast(mask, 'float32') 
        mask = K.expand_dims(mask, 2)  
        N = K.sum(mask, axis=1, keepdims=False)
        print(x.shape, mask.shape, N)
        mean = K.sum(x*mask, axis=1, keepdims=False)/N  
        #mean = K.sum(x, axis=1, keepdims=False)/N
        return mean


class GlobalMaskedAveragePooling1D_Spkr2(Layer):

    def __init__(self, **kwargs):
        super(GlobalMaskedAveragePooling1D_Spkr2, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        print("input shapes to masking layer is {}".format(input_shape))
        return (input_shape[0][0], input_shape[0][2])
    

    def call(self, inputs):
        x, SpkrID = inputs
        mask = K.equal(K.sum(K.abs(SpkrID), axis=2, keepdims=False), 2)  
        mask = K.cast(mask, 'float32') 
        mask = K.expand_dims(mask, 2)  
        N = K.sum(mask, axis=1, keepdims=False)
        print(x.shape, mask.shape, N)
        mean = K.sum(x*mask, axis=1, keepdims=False)/N  
        #mean = K.sum(x, axis=1, keepdims=False)/N
        return mean


class GlobalMaskedAveragePooling4D_to_3D_Conv1(Layer):

    def __init__(self, **kwargs):
        super(GlobalMaskedAveragePooling4D_to_3D_Conv1, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        print("input shapes to masking layer is {}".format(input_shape))
        return (input_shape[0][0], input_shape[0][2])
    

    def call(self, inputs):
        x, mask = inputs
        N = K.sum(mask, axis=1, keepdims=False)
        #print(x.shape, mask.shape, N)
        mask_shape = (mask.shape[1].value, mask.shape[2].value)
        print("Mask in masking_layers is {}".format(mask_shape))
        mask_1 = np.ones(mask_shape, dtype=int)
        mask_1[::2] = 0
        mask_2 = 1 - mask_1
        mask_1 = K.variable(mask_1)
        mask_2 = K.variable(mask_2)
        return  K.sum(x*mask*mask_1, axis=1, keepdims=False)/N

class GlobalMaskedAveragePooling4D_to_3D_Conv2(Layer):

    def __init__(self, **kwargs):
        super(GlobalMaskedAveragePooling4D_to_3D_Conv2, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        print("input shapes to masking layer is {}".format(input_shape))
        return (input_shape[0][0], input_shape[0][2]) 
    
    def call(self, inputs):
        x, mask = inputs
        N = K.sum(mask, axis=1, keepdims=False)
        #print(x.shape, mask.shape, N)
        mask_shape = (mask.shape[1].value, mask.shape[2].value)
        print("Mask in masking_layers is {}".format(mask_shape))
        mask_1 = np.ones(mask_shape, dtype=int)
        mask_1[::2] = 0
        mask_2 = 1 - mask_1
        mask_1 = K.variable(mask_1)
        mask_2 = K.variable(mask_2)
        return  K.sum(x*mask*mask_2, axis=1, keepdims=False)/N



