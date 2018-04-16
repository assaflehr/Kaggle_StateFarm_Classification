# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:02:12 2016

@author: Assaf
"""
from scipy.misc import imread, imresize
import numpy as np
import keras.preprocessing.image

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def load_transform(image_name):
    '''
        loads the file, returns it as 1xBGRx224x224 after mean-reduction
    '''
    img = imresize(imread(image_name, mode='RGB'), (224, 224)).astype(np.float32)
    #print img.shape
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    #print img.shape
    img = img.transpose((2, 0, 1))
    #print img.shape
    img = np.expand_dims(img, axis=0)  
    return img



    

def googlenet_augment(image_name,augmentation_strength=1,verbose=False):
    ''' augmentation_strength=1 do augmentation . 0=none.  1.5=stronger...'''
    def plotme():
        if verbose:
            plt.figure()
            plt.imshow(img)
            plt.show()
    
    img = imresize(imread(image_name, mode='RGB'), (224, 224)).astype(np.float32)
    if verbose: print 'load and resize' , img.shape
    plotme()
    
    if verbose: print 'reduce mean'
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    plotme()

    if verbose:print 'BGR RGB' , img.shape
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    plotme()
    
    if verbose:print 'random shift'
    wrg=0.1 * augmentation_strength
    hrg=0.1 * augmentation_strength
    img=keras.preprocessing.image.random_shift(img, wrg=wrg, hrg=hrg, row_index=0, col_index=1, channel_index=2)
    plotme()
    
    #if verbose:print 'random_rotation'
    #degree= 8* augmentation_strength
    #img=keras.preprocessing.image.random_rotation(img, degree, row_index=0, col_index=1, channel_index=2)
    #plotme()
    
    if verbose:print 'random_zoom'  #seem to change color on values [2,2]  [0.1,0.1] DO NOT USE
    level= 0.1 * augmentation_strength   #[1,1] means no zoom.  [0.5,0.5] is very strong.  Zoom-out not supported 
    img=keras.preprocessing.image.random_zoom(img,[1-level,1-level], row_index=0, col_index=1, channel_index=2)
    plotme()
    
    #def random_shear(x, intensity, row_index=1, col_index=2, channel_index=0,      fill_mode='nearest', cval=0.):
    #if verbose:print 'random_shear'
    #level = 0.2 * augmentation_strength
    #img=keras.preprocessing.image.random_shear(img,level,row_index=0, col_index=1, channel_index=2)
    #plotme()
    
    #print 'flip'
    #if random.random()<0.5:
    #    img=image.flip_axis(img, 1) # img_col_index=2
    #plotme()
    
    if verbose:print 'channel-change cant show no more'
    img = img.transpose((2, 0, 1))
    
    
    img =np.expand_dims(img, axis=0) 
    
    
    return img


class GoogleNetImageDataGenerator(ImageDataGenerator):
    ''' Uses a hack to change the keras.preprocessing.image.load_img to BGR. careful when using.
        googlenet expects the image format : float32, BatchxBGRxWxH (and not RGB) for example (1L, 3L, 224L, 224L)
        WxH=224x224, where values are G-= 116.779 R-= 123.68  B-= 103.939
    '''
    def load_img2(path, grayscale=False, target_size=None):
        img = imresize(imread(path, mode='RGB'), target_size).astype(np.float32)
        return img
    
    keras.preprocessing.image.load_img = load_img2
    
    '''overrides the BGR RGB'''
    def standardize(self, x):
       
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)
       
        #012 still RGB
        x[0,:,:] -= 123.68 # R
        x[1,:,:] -= 116.779 #G
        x[2,:,:] -= 103.939 #B
        #RGB to BGR
        x= x[::-1,:,:]
        #print 'standarize' , str(x.shape)  ,x.dtype
        
        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.zca_whitening:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
        #channel RGB to BGR
        return x
        


def check_equality():
    ''' compare the resut of googlenet when using various transforms
        first use the keras generator with our modified GoogleNetImageDataGenerator
        reading from '../../data/statefarm/imgs/cat
    '''
    from .create import load_partial_googlenet_model_weights
    from .create import create_googlenet
    from .create import GoogleNetClassifier
    from .create import googlenet_print_matches
    from keras.optimizers import SGD

    print 'cat.jpg'
    model = create_googlenet(GoogleNetClassifier(), weights=None)
    load_partial_googlenet_model_weights(model,'googlenet_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    print '####load_transform result'
    bottleneck_features_train= None
    #out is a list, as the number of the classifiers
    #out[2] is of size(#batch, 1000L)
    #so we need to iterate over out[2] and these are the results  out[2][0], out[2][1]...
    bottleneck_features_train = model.predict(load_transform('cat.jpg'))
    print type(bottleneck_features_train)
    print 'bottleneck_features_train[0].shape',bottleneck_features_train[0].shape
    probs= bottleneck_features_train[2][0]   
    googlenet_print_matches(probs)

    print '\n####GoogleNetImageDataGenerator:"'
    
    googlenet_cat_generator = GoogleNetImageDataGenerator().flow_from_directory(
            '../../data/statefarm/imgs/cat',
            target_size=(224, 224),
            batch_size=1,
            class_mode=None,
            shuffle=False)           
    bottleneck_features_train = model.predict_generator(googlenet_cat_generator, 1)
    
    def print_best(bottleneck_features_train):
       
        best_out = bottleneck_features_train[2]
        for i in range(best_out.shape[0]): #as the batch size
            probs=best_out[i]  #the first two auxilary classifer and the 3d , best one
            
            googlenet_print_matches(probs)
         #np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
    print_best(bottleneck_features_train)

    print '\n###now with googlenet_augment but augmentation_strength=0'
    bottleneck_features_train = model.predict(googlenet_augment('cat.jpg',augmentation_strength=0))
    print_best(bottleneck_features_train)
    
    #print '\nnow with googlenet_augment but augmentation_strength=0.2'
    #bottleneck_features_train = model.predict(googlenet_augment('cat.jpg',augmentation_strength=0.2))
    #print_best(bottleneck_features_train)

    print 'predicted'
    #285:Egyptian cat prob: 0.24039915204
    
def usage():
    image_name= 'cat3.jpg'#train_data_dir+'c1/img_1420.jpg'
    #googlenet_augment(image_name,augmentation_strength=1,verbose=True)
    #googlenet_augment(image_name,augmentation_strength=0.5,verbose=True)
    #googlenet_augment(image_name,augmentation_strength=0,verbose=True)
    
#    check_equality()
#USE TO CHECK RQUALITY THAT ALL
    


if __name__=='__main__':
    usage()
    