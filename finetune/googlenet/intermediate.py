# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 19:08:56 2016

@author: Assaf
"""

from finetune.googlenet.create import GoogleNetClassifier
import numpy as np

import os




class NoneClassifier(GoogleNetClassifier):
    
    def __init__(self):
        #GoogleNetClassifier.__init__(self)
        self.inception_4a_output=None
        self.inception_4d_output=None
        self.inception_5b_output=None   
    def branchOfAuxClassifer1(self, inception_4a_output):
        self.inception_4a_output = inception_4a_output

    def branchOfAuxClassifer2(self, inception_4d_output):
        self.inception_4d_output=inception_4d_output

    def mainClassfier(self, inception_5b_output):
        self.inception_5b_output = inception_5b_output
    
    def get_output(self):
        #print type(self.inception_4a_output)
        return [self.inception_4a_output,   self.inception_4d_output,  self.inception_5b_output]
    



def predict_outputs(model,batch_generator,out_root_folder,output_to_keep,size_to_predict):
    ''' runs on size_to_predict batches from the batch_generator, note that it's value is expected
        to be numpy-array of 1x1 with label , like [[5]] for folder 'c5'
        output_to_keep= when output is a list, which elements of it are important, for example
        in googlenet there are 3 output of aux0,aux1 and main classifiers . use [0,1,2] for all
        or only [2] for main output
        the output will be split to classifer folders (out_0/1/2) then class, then file
        out_0\c0\out_img_34.jpg.npy
             \c1\out_img_6.jpg.npy
        out_1\c0\out_img_34.jpg.npy
        etc
    '''
    from time import gmtime, strftime
    import time

    def build_folder_structure(root):
        '''  root/out_1/c0       root/out_1/c1      ...etc...   root/out_2/c0   ... '''
        for classifer in output_to_keep:
            classifer_folder= root +"/out_"+str(classifer)
            for i in range(0,10):
                label_folder = classifer_folder+'/c'+str(i)
                if not os.path.exists(label_folder):
                    os.makedirs(label_folder)

########## method 
    build_folder_structure ( out_root_folder)
    
    start_time=0
    for counter in range(size_to_predict):
        if counter==5:
            start_time = time.time()
        elif counter==15:
            elapsed = time.time()- start_time
            expected_for_all = elapsed * size_to_predict/10.0
            print 'For 10 took {:.2f} seconds. Predicted time for all {} is {:.2f} minutes'.format(elapsed,size_to_predict, expected_for_all/60.0)
            
        data_batch,label_batch,file_name = next(batch_generator)   # assumption batch_size=1 exactly!!!
        #PREDICT
        input_for_classifiers = model.predict(data_batch)
        #input_for_classifiers is now a list of 3 classifiers, 2 auxs and the final one
        
        #for i,input_i in enumerate(input_for_classifiers):
        for i in output_to_keep        :
            input_i = input_for_classifiers[i]
            label00 = label_batch[0][0]
            folder = out_root_folder + '/out_'+str(i) +'/c'+ str(label00) +'/'
            np.save(open(folder+'out_{}.npy'.format(file_name[0]), 'wb'), input_i)
            if counter==0:
                print 'output for classifier',i , input_i.shape
            elif counter%100==0:
                print counter, 'completed',strftime("_%Y-%m-%d_%H-%M-%S", gmtime())
                




def test_loading_from_binary(no_head_model):
    '''
        predict_from_mem() runs an image->no_head_model->output->classifier_head
        predict_from_file() loads a numpy file (the previous output) and loaded_output->classifer_head
    '''
    from finetune.googlenet.create import load_partial_googlenet_model_weights
    from finetune.googlenet.create import googlenet_print_matches
    
    #now head model
    inception_5b_output = Input(shape=(1024,7,7))
    pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7),strides=(1,1),name='pool5/7x7_s2')(inception_5b_output)
    loss3_flat = Flatten()(pool5_7x7_s1)
    pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)
    loss3_classifier = Dense(1000,name='loss3/classifier',W_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
    loss3_classifier_act = Activation('softmax',name='prob')(loss3_classifier)
    head_model =  m= Model(input=inception_5b_output,output=loss3_classifier_act)
    
    print ('\n\nloading head_model')
    load_partial_googlenet_model_weights(head_model,'googlenet_weights.h5')
    head_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    
    def predict_from_mem():
        print ('\n\npredict_from_mem')
        data, label, files = next(from_train_generator)
        print data.shape, label.shape, label, files
        no_head_outputs = no_head_model.predict(data)
        print 'no_head_outputs[2].shape', no_head_outputs[2].shape
        head_output   = head_model.predict( no_head_outputs[2])
        googlenet_print_matches(head_output[0]) #looks good >0.5 of seatbelt
        
       
        file_name= 'predict_from_mem.npy'
        saved = no_head_outputs[2]
        np.save(open(file_name, 'wb'), saved)  #note b is needed in windows as text-file on windows are altered by python
        loaded = np.load(open(file_name,'rb'))
        # saved should be ==loaded
        head_output   = head_model.predict( loaded)
        googlenet_print_matches(head_output[0]) #looks good >0.5 of seatbelt
    
    def predict_from_file():
        print ('\n\npredict_from_file')
        from_googlenet_generator= generate_data_and_label(PATH_TO_IMGS+'googlenet/train/out_2' ,
                                  load_transform_method= lambda filename: np.load(open(filename,"rb")),
                                  batch_size=1,shuffle=True,verbose=False) 
        data, label ,files= next(from_googlenet_generator)
        print data.shape, label.shape, label,files
        no_head_outputs = data
        output = head_model.predict(no_head_outputs)
        googlenet_print_matches(output[0])
    predict_from_mem()
    predict_from_file()
    

#usage()

            
def usage():
    import time
    import random
    from finetune.googlenet.create import create_googlenet
    from finetune.googlenet.create import load_partial_googlenet_model_weights
    from finetune.generator import generate_data_and_label


    PATH_TO_IMGS='C:/code/git/data/statefarm/imgs/'
    print 'creating headless googlenet'
    no_head_model = create_googlenet(NoneClassifier(), weights=None)
    load_partial_googlenet_model_weights(no_head_model,'googlenet_weights.h5')
    no_head_model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    from finetune.googlenet.transform import load_transform
    from finetune.googlenet.transform import googlenet_augment
    
    from_train_generator= generate_data_and_label(PATH_TO_IMGS+'train',
                                           load_transform,
                                           lambda folder: [folder[1:]], #'c3'-> '3',
                                           file_name_transform=lambda fullname: os.path.split(fullname)[1]+str(random.randint(0,10000))
                                          + str(time.time()),
                                           batch_size=1,shuffle=False,verbose=False)    
    predict_outputs(no_head_model, from_train_generator,     PATH_TO_IMGS+'googlenet/train'     ,[1,2],19390)  #19390  
    from_train_generator= generate_data_and_label(PATH_TO_IMGS+'train',
                                           googlenet_augment,
                                           lambda folder: [folder[1:]], #'c3'-> '3',
                                           file_name_transform=lambda fullname: os.path.split(fullname)[1]+str(random.randint(0,10000))
                                           + str(time.time()),
                                           batch_size=1,shuffle=False,verbose=False)    
    predict_outputs(no_head_model, from_train_generator,     PATH_TO_IMGS+'googlenet/train_aug'     ,[1,2],2*19390)  #19390  
    
    
    from_validation_generator= generate_data_and_label(PATH_TO_IMGS+'validation',
                                           load_transform,
                                           lambda folder: [folder[1:]],
                                           batch_size=1,shuffle=False,verbose=False) 
    #predict_outputs(no_head_model, from_validation_generator,PATH_TO_IMGS+'googlenet/validation',[2],3034)  #3034
    #test_loading_from_binary(no_head_model)

if __name__=='__main__':
   
  # print 'For 10 took {0:.2f} seconds. Predicted time for all {1} is {2} minutes'.format(0.100001,0.1,0.1)
            
    usage()



                                                                                     
