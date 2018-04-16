# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:23:52 2016

@author: Assaf
"""
from keras.layers import Input, Dense , Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Activation
from keras.models import Model
from keras.regularizers import l2

'''
def branchOfAuxClassifer0(includeInput=None): 
    if (includeInput):
        inception_4a_output=includeInput
    else:
        inception_4a_output= Input(shape=(512, 14, 14))
    loss1_ave_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),name='loss1/ave_pool')(inception_4a_output)
    loss1_conv = Convolution2D(128,1,1,border_mode='same',activation='relu',name='loss1/conv',W_regularizer=l2(0.0002))(loss1_ave_pool)
    loss1_flat = Flatten()(loss1_conv)
    loss1_fc = Dense(1024,activation='relu',name='loss1/fc',W_regularizer=l2(0.0002))(loss1_flat)
    loss1_drop_fc = Dropout(0.7)(loss1_fc)
    loss1_classifier = Dense(10,name='loss1/classifier',W_regularizer=l2(0.0002))(loss1_drop_fc)
    loss1_classifier_act = Activation('softmax',name='prob_aux0')(loss1_classifier)
    
    if includeInput:
        return loss1_classifier_act
    else:
        return Model(input=inception_4a_output, output=loss1_classifier_act)

def branchOfAuxClassifer1(includeInput=None): 
    if (includeInput):
        inception_4d_output=includeInput
    else:
        inception_4d_output  = Input(shape=(528, 14, 14))
   
    loss2_ave_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),name='loss2/ave_pool')(inception_4d_output)
    loss2_conv = Convolution2D(128,1,1,border_mode='same',activation='relu',name='loss2/conv',W_regularizer=l2(0.0002))(loss2_ave_pool)
    loss2_flat = Flatten()(loss2_conv)
    loss2_fc = Dense(1024,activation='relu',name='loss2/fc',W_regularizer=l2(0.0002))(loss2_flat)
    loss2_drop_fc = Dropout(0.7)(loss2_fc)
    loss2_classifier = Dense(10,name='loss2/classifier',W_regularizer=l2(0.0002))(loss2_drop_fc)
    loss2_classifier_act = Activation('softmax',name='prob_aux1')(loss2_classifier)   
    if includeInput:
        return loss2_classifier_act
    else:
        return Model(input=inception_4d_output, output=loss2_classifier_act)
        
        

def mainClassfier2(includeInput=None):  #2
    if (includeInput):
        inception_5b_output = includeInput
    else: 
        inception_5b_output = Input(shape=(1024,7,7))
    pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7),strides=(1,1),name='pool5/7x7_s2')(inception_5b_output)
    loss3_flat = Flatten()(pool5_7x7_s1)
    pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)
    loss3_classifier = Dense(10,name='loss3/classifier',W_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
    loss3_classifier_act = Activation('softmax',name='prob')(loss3_classifier)
    
    if (includeInput):
        return loss3_classifier_act
    else:
        return  Model(input=inception_5b_output,output=loss3_classifier_act)
    
'''
from finetune.generator import generate_data_and_label
from finetune.plot  import graph_history
from finetune.persistency import saveToFiles
import numpy as np
from keras.optimizers import SGD
from finetune.googlenet.create import create_googlenet
from finetune.googlenet.create import load_partial_googlenet_model_weights


def classifier_head(input):
    ''' create a functional (not sequential) of a classifier-head, apply it on input parameter
        return a variable pointing to the output of the classifier (the caller can call Model(...,output=<returned_val>))
    '''
    pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7),strides=(1,1),name='pool5/7x7_s2')(input)
    loss3_flat = Flatten()(pool5_7x7_s1)
    pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)
    loss3_classifier = Dense(10,name='loss3/classifier',W_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
    loss3_classifier_act = Activation('softmax',name='prob')(loss3_classifier)
    return loss3_classifier_act
   

def create_full_model_only_aux0(filename):
    nc=NoneClassifier()
    no_head_model = create_googlenet(nc, weights=None)
    load_partial_googlenet_model_weights(no_head_model,'googlenet_weights.h5')
    
    outputs = [branchOfAuxClassifer0(nc.inception_4a_output)  ]
    full_model_aux0 = Model(input=no_head_model.input, output=outputs) #maybe change this to model.outut?
    load_partial_googlenet_model_weights(full_model_aux0,filename,verbose=False)
    return full_model_aux0


def create_full_model_3_classifiers():
    model = None
    nc=NoneClassifier()
    no_head_model = create_googlenet(nc, weights=None)
    load_partial_googlenet_model_weights(no_head_model,'googlenet_weights.h5')
    no_head_model_layers= dict(zip([layer.name for layer in no_head_model.layers],no_head_model.layers))
    print 'no_head_model loaded'


    #TODO retrain model_chapter6_aux1, I just invented here something....
    head_model_h5_files = ['current/model_chapter6_aux0_25epoc.h5',
                           'current/model_chapter6_aux1_try2_11epoc.h5',
                           'current/model_chapter6_12epoc.h5'
                          ]
    classifier_func = [ branchOfAuxClassifer0, 
                        branchOfAuxClassifer1, 
                        mainClassfier2
                      ]

    no_head_connection_vars= [nc.inception_4a_output ,
                              nc.inception_4d_output,
                              nc.inception_5b_output
                             ]

    outputs=[]
    for i in range(len(head_model_h5_files)):
        output = classifier_func[i](no_head_connection_vars[i])  
        outputs.append(output)

    full_model = Model(input=no_head_model.input, output=outputs) #maybe change this to model.outut?
    print 'model defined'

    #load wieghts:  TODO: check no naming collision here, or missing name
    for i in range(len(head_model_h5_files)):
        print i, 'loading file ',head_model_h5_files[i]
        load_partial_googlenet_model_weights(full_model,head_model_h5_files[i],verbose=False)
    print 'loaded pre-trained'
    return full_model
    
 
def usage2():
    
    in_layer = Input(shape=(1024,7,7)) #size of the output binary file
    model = Model(input= in_layer,output=classifier_head(in_layer) )
    model.compile(optimizer='rmsprop',  loss='categorical_crossentropy',    metrics=['accuracy'])
     
    PATH_TO_IMGS='C:/code/git/data/statefarm/imgs/'
    from_train_generator= generate_data_and_label(PATH_TO_IMGS+'googlenet/train/out_2' ,  #input folder
                                       load_transform_method= lambda filename: np.load(open(filename,"rb")),
                                       file_name_transform=None,batch_size=64,shuffle=True,verbose=False) 
    from_valid_generator= generate_data_and_label(PATH_TO_IMGS+'googlenet/validation/out_2' ,
                                       load_transform_method= lambda filename: np.load(open(filename,"rb")),
                                       file_name_transform=None,batch_size=64,shuffle=True,verbose=False)  
    history_list=[]    
    print 'hererere'
    for i in range (1,2):
        print i
        history=model.fit_generator(from_train_generator, samples_per_epoch=1000, nb_epoch=1, verbose=1,
                            callbacks=[], validation_data=from_valid_generator, nb_val_samples=100, class_weight=None, max_q_size=3) 
        history_list.append(history)
        graph_history(history_list)
        if i>1 and i%5==0:
            graph_history(history_list)
            saveToFiles(model,'train_classifier_out_2_epoc'+str(i))
        
    print 'done1'
    print 'done2'    
