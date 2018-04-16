# -*- coding: utf-8 -*-


from keras.layers.core import Layer
import theano.tensor as T



class LRN(Layer):

    def __init__(self, alpha=0.0001,k=1,beta=0.75,n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        b, ch, r, c = x.shape
        half_n = self.n // 2 # half the local region
        input_sqr = T.sqr(x) # square the input
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c) # make an empty tensor with zero pads along channel dimension
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],input_sqr) # set the center to be the squared input
        scale = self.k # offset for the scale
        norm_alpha = self.alpha / self.n # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolHelper(Layer):
    
    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        return x[:,:,1:,1:]
    
    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
class GoogleNetClassifier:
    ''' Allows to change the classifer layer and the output of the google-net easily, by subclassing
        override get_otuput() to get different output, for example the inception_5b_output, which can be used to record layer.
        override branchOfAuxClassifer1/2,branchOfAuxClassifer2 to create different classifers
    '''
    loss1_classifier_act=None
    loss2_classifier_act=None
    loss3_classifier_act=None
    
    inception_4a_output=None
    inception_4d_output=None
    inception_5b_output=None
    
    def branchOfAuxClassifer1(self, inception_4a_output):
      
        self.inception_4a_output=inception_4a_output
        loss1_ave_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),name='loss1/ave_pool')(inception_4a_output)

        loss1_conv = Convolution2D(128,1,1,border_mode='same',activation='relu',name='loss1/conv',W_regularizer=l2(0.0002))(loss1_ave_pool)

        loss1_flat = Flatten()(loss1_conv)

        loss1_fc = Dense(1024,activation='relu',name='loss1/fc',W_regularizer=l2(0.0002))(loss1_flat)

        loss1_drop_fc = Dropout(0.7)(loss1_fc)

        loss1_classifier = Dense(1000,name='loss1/classifier',W_regularizer=l2(0.0002))(loss1_drop_fc)

        self.loss1_classifier_act = Activation('softmax')(loss1_classifier)

    def branchOfAuxClassifer2(self, inception_4d_output):
        self.inception_4d_output = inception_4d_output
        loss2_ave_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),name='loss2/ave_pool')(inception_4d_output)

        loss2_conv = Convolution2D(128,1,1,border_mode='same',activation='relu',name='loss2/conv',W_regularizer=l2(0.0002))(loss2_ave_pool)

        loss2_flat = Flatten()(loss2_conv)

        loss2_fc = Dense(1024,activation='relu',name='loss2/fc',W_regularizer=l2(0.0002))(loss2_flat)

        loss2_drop_fc = Dropout(0.7)(loss2_fc)

        loss2_classifier = Dense(1000,name='loss2/classifier',W_regularizer=l2(0.0002))(loss2_drop_fc)

        self.loss2_classifier_act = Activation('softmax')(loss2_classifier)

    def mainClassfier(self, inception_5b_output):
        self.inception_5b_output = inception_5b_output
        pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7),strides=(1,1),name='pool5/7x7_s2')(inception_5b_output)

        loss3_flat = Flatten()(pool5_7x7_s1)

        pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)

        loss3_classifier = Dense(1000,name='loss3/classifier',W_regularizer=l2(0.0002))(pool5_drop_7x7_s1)

        self.loss3_classifier_act = Activation('softmax',name='prob')(loss3_classifier)
   
    def get_output(self):
        return [self.loss1_classifier_act,self.loss2_classifier_act,self.loss3_classifier_act]
    
    

import numpy as np
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
#from googlenet_custom_layers import PoolHelper,LRN
#see googlenet here: https://arxiv.org/pdf/1409.4842v1.pdf

def create_googlenet(classifier,weights=None):
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    # https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14
    input = Input(shape=(3, 224, 224))
    
    conv1_7x7_s2 = Convolution2D(64,7,7,subsample=(2,2),border_mode='same',activation='relu',name='conv1/7x7_s2',W_regularizer=l2(0.0002))(input)
    
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)
    
    pool1_helper = PoolHelper()(conv1_zero_pad)
    
    pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool1/3x3_s2')(pool1_helper)
    
    #GREEN 1
    pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)
    
    conv2_3x3_reduce = Convolution2D(64,1,1,border_mode='same',activation='relu',name='conv2/3x3_reduce',W_regularizer=l2(0.0002))(pool1_norm1)
    
    conv2_3x3 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='conv2/3x3',W_regularizer=l2(0.0002))(conv2_3x3_reduce)
    
    #GREEN 2
    conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)
    
    conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
    
    pool2_helper = PoolHelper()(conv2_zero_pad)
    
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool2/3x3_s2')(pool2_helper)
    
    
    inception_3a_1x1 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_3a/1x1',W_regularizer=l2(0.0002))(pool2_3x3_s2)
    
    inception_3a_3x3_reduce = Convolution2D(96,1,1,border_mode='same',activation='relu',name='inception_3a/3x3_reduce',W_regularizer=l2(0.0002))(pool2_3x3_s2)
    
    inception_3a_3x3 = Convolution2D(128,3,3,border_mode='same',activation='relu',name='inception_3a/3x3',W_regularizer=l2(0.0002))(inception_3a_3x3_reduce)
    
    inception_3a_5x5_reduce = Convolution2D(16,1,1,border_mode='same',activation='relu',name='inception_3a/5x5_reduce',W_regularizer=l2(0.0002))(pool2_3x3_s2)
    
    inception_3a_5x5 = Convolution2D(32,5,5,border_mode='same',activation='relu',name='inception_3a/5x5',W_regularizer=l2(0.0002))(inception_3a_5x5_reduce)
    
    inception_3a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_3a/pool')(pool2_3x3_s2)
    
    inception_3a_pool_proj = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_3a/pool_proj',W_regularizer=l2(0.0002))(inception_3a_pool)
    
    #Green 3  (3a)
    inception_3a_output = merge([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj],mode='concat',concat_axis=1,name='inception_3a/output')
    
    
    inception_3b_1x1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_3b/1x1',W_regularizer=l2(0.0002))(inception_3a_output)
    
    inception_3b_3x3_reduce = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_3b/3x3_reduce',W_regularizer=l2(0.0002))(inception_3a_output)
    
    inception_3b_3x3 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='inception_3b/3x3',W_regularizer=l2(0.0002))(inception_3b_3x3_reduce)
    
    inception_3b_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_3b/5x5_reduce',W_regularizer=l2(0.0002))(inception_3a_output)
    
    inception_3b_5x5 = Convolution2D(96,5,5,border_mode='same',activation='relu',name='inception_3b/5x5',W_regularizer=l2(0.0002))(inception_3b_5x5_reduce)
    
    inception_3b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_3b/pool')(inception_3a_output)
    
    inception_3b_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_3b/pool_proj',W_regularizer=l2(0.0002))(inception_3b_pool)
    
    #GREEN 4  (3b)
    inception_3b_output = merge([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj],mode='concat',
                                concat_axis=1,name='inception_3b/output' )
    
    
    inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)
    
    pool3_helper = PoolHelper()(inception_3b_output_zero_pad)
    
    pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool3/3x3_s2')(pool3_helper)
    
    
    inception_4a_1x1 = Convolution2D(192,1,1,border_mode='same',activation='relu',name='inception_4a/1x1',W_regularizer=l2(0.0002))(pool3_3x3_s2)
    
    inception_4a_3x3_reduce = Convolution2D(96,1,1,border_mode='same',activation='relu',name='inception_4a/3x3_reduce',W_regularizer=l2(0.0002))(pool3_3x3_s2)
    
    inception_4a_3x3 = Convolution2D(208,3,3,border_mode='same',activation='relu',name='inception_4a/3x3',W_regularizer=l2(0.0002))(inception_4a_3x3_reduce)
    
    inception_4a_5x5_reduce = Convolution2D(16,1,1,border_mode='same',activation='relu',name='inception_4a/5x5_reduce',W_regularizer=l2(0.0002))(pool3_3x3_s2)
    
    inception_4a_5x5 = Convolution2D(48,5,5,border_mode='same',activation='relu',name='inception_4a/5x5',W_regularizer=l2(0.0002))(inception_4a_5x5_reduce)
    
    inception_4a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4a/pool')(pool3_3x3_s2)
    
    inception_4a_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4a/pool_proj',W_regularizer=l2(0.0002))(inception_4a_pool)
    
    #GREEN 5  (end of 4a)
    inception_4a_output = merge([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj],mode='concat',concat_axis=1,name='inception_4a/output')
    
    
    
    classifier.branchOfAuxClassifer1(inception_4a_output)
    
    
    inception_4b_1x1 = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_4b/1x1',W_regularizer=l2(0.0002))(inception_4a_output)
    
    inception_4b_3x3_reduce = Convolution2D(112,1,1,border_mode='same',activation='relu',name='inception_4b/3x3_reduce',W_regularizer=l2(0.0002))(inception_4a_output)
    
    inception_4b_3x3 = Convolution2D(224,3,3,border_mode='same',activation='relu',name='inception_4b/3x3',W_regularizer=l2(0.0002))(inception_4b_3x3_reduce)
    
    inception_4b_5x5_reduce = Convolution2D(24,1,1,border_mode='same',activation='relu',name='inception_4b/5x5_reduce',W_regularizer=l2(0.0002))(inception_4a_output)
    
    inception_4b_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4b/5x5',W_regularizer=l2(0.0002))(inception_4b_5x5_reduce)
    
    inception_4b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4b/pool')(inception_4a_output)
    
    inception_4b_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4b/pool_proj',W_regularizer=l2(0.0002))(inception_4b_pool)
    #GREEN 6  (end of 4b)
    inception_4b_output = merge([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj],mode='concat',concat_axis=1,name='inception_4b_output')
    
    
    inception_4c_1x1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4c/1x1',W_regularizer=l2(0.0002))(inception_4b_output)
    
    inception_4c_3x3_reduce = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4c/3x3_reduce',W_regularizer=l2(0.0002))(inception_4b_output)
    
    inception_4c_3x3 = Convolution2D(256,3,3,border_mode='same',activation='relu',name='inception_4c/3x3',W_regularizer=l2(0.0002))(inception_4c_3x3_reduce)
    
    inception_4c_5x5_reduce = Convolution2D(24,1,1,border_mode='same',activation='relu',name='inception_4c/5x5_reduce',W_regularizer=l2(0.0002))(inception_4b_output)
    
    inception_4c_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4c/5x5',W_regularizer=l2(0.0002))(inception_4c_5x5_reduce)
    
    inception_4c_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4c/pool')(inception_4b_output)
    
    inception_4c_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4c/pool_proj',W_regularizer=l2(0.0002))(inception_4c_pool)
     #GREEN 7  (end of 4c)
    inception_4c_output = merge([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj],mode='concat',concat_axis=1,name='inception_4c/output')
    
    
    inception_4d_1x1 = Convolution2D(112,1,1,border_mode='same',activation='relu',name='inception_4d/1x1',W_regularizer=l2(0.0002))(inception_4c_output)
    
    inception_4d_3x3_reduce = Convolution2D(144,1,1,border_mode='same',activation='relu',name='inception_4d/3x3_reduce',W_regularizer=l2(0.0002))(inception_4c_output)
    
    inception_4d_3x3 = Convolution2D(288,3,3,border_mode='same',activation='relu',name='inception_4d/3x3',W_regularizer=l2(0.0002))(inception_4d_3x3_reduce)
    
    inception_4d_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_4d/5x5_reduce',W_regularizer=l2(0.0002))(inception_4c_output)
    
    inception_4d_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4d/5x5',W_regularizer=l2(0.0002))(inception_4d_5x5_reduce)
    
    inception_4d_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4d/pool')(inception_4c_output)
    
    inception_4d_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4d/pool_proj',W_regularizer=l2(0.0002))(inception_4d_pool)
    #GREEN 8  (end of 4d)
    inception_4d_output = merge([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj],mode='concat',concat_axis=1,name='inception_4d/output')
    
   
    ########
    classifier.branchOfAuxClassifer2(inception_4d_output)
    
    
    inception_4e_1x1 = Convolution2D(256,1,1,border_mode='same',activation='relu',name='inception_4e/1x1',W_regularizer=l2(0.0002))(inception_4d_output)
    
    inception_4e_3x3_reduce = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_4e/3x3_reduce',W_regularizer=l2(0.0002))(inception_4d_output)
    
    inception_4e_3x3 = Convolution2D(320,3,3,border_mode='same',activation='relu',name='inception_4e/3x3',W_regularizer=l2(0.0002))(inception_4e_3x3_reduce)
    
    inception_4e_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_4e/5x5_reduce',W_regularizer=l2(0.0002))(inception_4d_output)
    
    inception_4e_5x5 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_4e/5x5',W_regularizer=l2(0.0002))(inception_4e_5x5_reduce)
    
    inception_4e_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4e/pool')(inception_4d_output)
    
    inception_4e_pool_proj = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4e/pool_proj',W_regularizer=l2(0.0002))(inception_4e_pool)
    #GREEN
    inception_4e_output = merge([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj],mode='concat',concat_axis=1,name='inception_4e/output')
    
    
    inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)
    
    pool4_helper = PoolHelper()(inception_4e_output_zero_pad)
    
    pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool4/3x3_s2')(pool4_helper)
    
    
    inception_5a_1x1 = Convolution2D(256,1,1,border_mode='same',activation='relu',name='inception_5a/1x1',W_regularizer=l2(0.0002))(pool4_3x3_s2)
    
    inception_5a_3x3_reduce = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_5a/3x3_reduce',W_regularizer=l2(0.0002))(pool4_3x3_s2)
    
    inception_5a_3x3 = Convolution2D(320,3,3,border_mode='same',activation='relu',name='inception_5a/3x3',W_regularizer=l2(0.0002))(inception_5a_3x3_reduce)
    
    inception_5a_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_5a/5x5_reduce',W_regularizer=l2(0.0002))(pool4_3x3_s2)
    
    inception_5a_5x5 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_5a/5x5',W_regularizer=l2(0.0002))(inception_5a_5x5_reduce)
    
    inception_5a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_5a/pool')(pool4_3x3_s2)
    
    inception_5a_pool_proj = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_5a/pool_proj',W_regularizer=l2(0.0002))(inception_5a_pool)
    #GREEN
    inception_5a_output = merge([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj],mode='concat',concat_axis=1,name='inception_5a/output')
    
    
    inception_5b_1x1 = Convolution2D(384,1,1,border_mode='same',activation='relu',name='inception_5b/1x1',W_regularizer=l2(0.0002))(inception_5a_output)
    
    inception_5b_3x3_reduce = Convolution2D(192,1,1,border_mode='same',activation='relu',name='inception_5b/3x3_reduce',W_regularizer=l2(0.0002))(inception_5a_output)
    
    inception_5b_3x3 = Convolution2D(384,3,3,border_mode='same',activation='relu',name='inception_5b/3x3',W_regularizer=l2(0.0002))(inception_5b_3x3_reduce)
    
    inception_5b_5x5_reduce = Convolution2D(48,1,1,border_mode='same',activation='relu',name='inception_5b/5x5_reduce',W_regularizer=l2(0.0002))(inception_5a_output)
    
    inception_5b_5x5 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_5b/5x5',W_regularizer=l2(0.0002))(inception_5b_5x5_reduce)
    
    inception_5b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_5b/pool')(inception_5a_output)
    
    inception_5b_pool_proj = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_5b/pool_proj',W_regularizer=l2(0.0002))(inception_5b_pool)
    #GREEN
    inception_5b_output = merge([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj],mode='concat',concat_axis=1,name='inception_5b/output')
    
    classifier.mainClassfier(inception_5b_output)


    googlenet = Model(input=input, output=classifier.get_output() )
    

    
    return googlenet



import h5py
import os
def load_partial_googlenet_model_weights(model, weights_path,verbose=False):
    '''
        Be adviced that it's tailored for specific  googlenet_weights.h5 file 
        and will not work for other models, even if it's googlenet too
    '''
    if verbose:
        print weights_path
    assert os.path.exists(weights_path)
    #our model does not equal googlenet.  It lack ~10 layers related to classifer. lets load weights of the rest
    layerByName= dict(zip([layer.name for layer in model.layers],model.layers))
   
    f = h5py.File(weights_path)
    if verbose:
        print 'weights_path',f.items()
        
    # an h5 file is has items which can be data or groups(==like folders), which can be recursive 
    #see h5 tutorial https://www.getdatajoy.com/learn/Read_and_Write_HDF5_from_Python
    '''
     conv1 <HDF5 group "/conv1" (1 members)>
        7x7_s2 <HDF5 group "/conv1/7x7_s2" (1 members)>
            conv1 <HDF5 group "/conv1/7x7_s2/conv1" (2 members)>
                7x7_s2_W <HDF5 dataset "7x7_s2_W": shape (64, 3, 7, 7), type "<f4">
                7x7_s2_b <HDF5 dataset "7x7_s2_b": shape (64,), type "<f4">
     conv2 <HDF5 group "/conv2" (3 members)>
        3x3 <HDF5 group "/conv2/3x3" (1 members)>
            conv2 <HDF5 group "/conv2/3x3/conv2" (2 members)>
                3x3_W <HDF5 dataset "3x3_W": shape (192, 64, 3, 3), type "<f4">
                3x3_b <HDF5 dataset "3x3_b": shape (192,), type "<f4">
        3x3_reduce <HDF5 group "/conv2/3x3_reduce" (1 members)>
            conv2 <HDF5 group "/conv2/3x3_reduce/conv2" (2 members)>
                3x3_reduce_W <HDF5 dataset "3x3_reduce_W": shape (64, 64, 1, 1), type "<f4">
                3x3_reduce_b <HDF5 dataset "3x3_reduce_b": shape (64,), type "<f4">
        norm2 <HDF5 group "/conv2/norm2" (0 members)>
    '''   
    if verbose:
        print 'weights_path',f.items()
    for name1,group1 in f.items():
        if verbose: print name1 , group1
        for name2,group2 in group1.items():
            if verbose: print '\t',name2 , group2
            for name3,group3 in group2.items():
                if verbose: print '\t\t',name3 , group3
                path = name1+'/'+name2
                if path in layerByName:
                    weightsList = [group3.get(weight_name) for weight_name in group3.keys()]
                    if verbose: print '\t\tLOADED WEIGHTS for ',path                    
                    layerByName[path].set_weights(weightsList)
                   
                #for name4,group4 in group3.items():
                #    print '\t\t\t',name4 , group4
    f.close()
    

            


    
    

def googlenet_print_matches(probs):
    from ..imagenet.labels import imagenet_mapping
    best_labels=  probs.argsort()[-5:][::-1]
    for label in best_labels: 
            print str(label)+":"+ imagenet_mapping[label],'prob:',str(probs[label])
            

def usage():
    from .transform import load_transform
    #from ..imagenet.labels import imagenet_mapping
    #img = imresize(imread('cat.jpg', mode='RGB'), (224, 224)).astype(np.float32)
    image_names= ["cat.jpg",
             "cat2.jpg"
             #"../../data/statefarm/imgs/train/c3/img_5.jpg",
             
            # "../../data/statefarm/imgs/train/c3/img_204.jpg"
            ]
    model = create_googlenet(GoogleNetClassifier())
    load_partial_googlenet_model_weights(model,'googlenet_weights.h5')
    #model.load_weights('googlenet_weights.h5')
    print 'defined model and loaded weights'
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    print 'compiled model'
    
    
    X = np.concatenate([load_transform(name) for name in image_names],axis=0)
   
        
        
    print X.shape  #(2L, 3L, 224L, 224L)
    print '\ntransformed image'
    # Test pretrained model

    model.predict(X) # note: the model has three outputs
    model.predict(X) # note: the model has three outputs
    out = model.predict(X)
    #out is a list, as the number of the classifiers
    #out[2] is of size(#batch, 1000L)
    #so we need to iterate over out[2] and these are the results  out[2][0], out[2][1]...
    probs= out[2][0]   #the first two auxilary classifer and the 3d , best one
    googlenet_print_matches(probs)
    print '\n'
    probs= out[2][1]   #the first two auxilary classifer and the 3d , best one
    googlenet_print_matches(probs)
        
#  for cat:
      #  285:Egyptian cat prob: 0.35569486022
      #  700:paper towel prob: 0.0903356820345
      #  281:tabby, tabby cat prob: 0.0791673213243
      #  287:lynx, catamount prob: 0.0361604392529
      #  282:tiger cat prob: 0.0346178822219
#img_5.jpg
    #785:seat belt, seatbelt prob: 0.588668346405
    #475:car mirror prob: 0.11258764565
        
if __name__ == "__main__":
    #when runnng it on spyter, use F6
    #run from: keras folder, command line: -m finetune.googlenet.defs
    print __package__
    usage()
    