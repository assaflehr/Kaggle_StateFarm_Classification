# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 18:13:49 2016

@author: Assaf
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np




def print_report(classes_list, all_probs,label_mapping=None):
    ''' classes_list = [9, 5, 1,]'''
    #see http://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
    y_true =  [int(c) for c in classes_list]
    y_pred = [np.argmax(probs) for probs in all_probs]
    #print y_true
    #print y_pred
    c_matrix = confusion_matrix(y_true, y_pred)
    print c_matrix
    
  
    target_names = [str(i)+' '+(label_mapping['c'+str(i)] if label_mapping else str(i)) for i in range(0,10)]
    print(classification_report(y_true, y_pred, target_names=target_names))





import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

pred = [(0.5,0.5)]      
act = [(1,0)]
#print (logloss((1,0), (0.5,0.5))) #0.69
#print (logloss((0,1), (1,0)))     #very wrong- 34.5
#print (logloss((1,0), (1,0)))     # true - ~0 (9e-16)
#print (logloss((1,0,0,0), (0,1,1,1))) # 0.1

def score_batch(actual_class_list,pred_nparray,verbose=0):
    '''
        return score and misses indexes loss bigger than 0.1 on one item
    '''
    #create an array from one class
    print 'calculating score of' , pred_nparray.shape[0], 'results'
    if len(actual_class_list)!=pred_nparray.shape[0] :
        raise ValueError('wrong shapes')
    scores=[]
    miss_scores=[]
    miss_index=[]
    for i,c in enumerate(actual_class_list):
        act_list = [ 0  for _ in range(0,pred_nparray.shape[1])]
        act_list[int(c)]=1
        
        pred_list = pred_nparray[i].tolist()
        curr_score=logloss(act_list,pred_list )
        scores.append(curr_score)
        if curr_score>0.1:
            miss_index.append(i)
            miss_scores.append(curr_score)
            if verbose:
                print i,'WRONG',curr_score #,act_list,pred_list
        elif verbose:
                print i,'RIGHT\t\t\t',curr_score #,act_list,pred_list
       
    if verbose:
        print 'median'  ,np.median(np.array(scores))
        print 'mean',np.mean(np.array(scores))
        print 'misses >0.1 mean',np.mean(np.array(miss_scores))
        print 'misses >0.1 median',np.median(np.array(miss_scores))
    
    return sum(scores)/pred_nparray.shape[0],miss_index
 
 
 
PATH_TO_DATA = '../../data/statefarm/'
PATH_TO_IMGS = PATH_TO_DATA + 'imgs/'
train_data_dir = PATH_TO_IMGS+'train/'
validation_data_dir = PATH_TO_IMGS+'/validation/'
test_data_dir = PATH_TO_IMGS+'/test/'






from finetune.generator import predict_directory
from finetune.generator import folder_to_label


def predict_val_and_score(m,classifer_index,load_transform_method,samples=None):
    #model_name='aux0'
    #m=create_full_model_only_aux0()
    c_name_list,all_probs= predict_directory(m,validation_data_dir,
                                                 load_transform_method= load_transform_method,
                                                 transform_folder_to_label=folder_to_label,
                                                 file_name_transform=lambda fullname: fullname.split('/')[-2:],
                                                 classifer_index=classifer_index,
                                                 samples=samples)
    classes_list = [x[0][1:] for x in c_name_list]
                                                       
    score,miss_indexes= score_batch(classes_list,all_probs,False)
    accuracy = 1- float(len(miss_indexes))/ len(c_name_list)
    print '\nSCORE=',score,'accuracy=',accuracy
    #print 'misses_index' , miss_indexes , 'of total',len(c_name_list)
    print 'first 50 misses',[c_name_list[i]  for i in miss_indexes[:50]]
    return classes_list,all_probs


from time import gmtime, strftime
def time_str(): 
    return strftime("_%Y-%m-%d_%H-%M", gmtime())


#######################################################################################
#######################################################################################
import pandas
import os
def predict_to_csv(filenameNoPrefix,pathname_list,all_probs):
    
    # The expected out file should be:
    #img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9
    #img_0.jpg,1,0,0,0,0,...,0
    #img_1.jpg,0.3,0.1,0.6,0,...,0
    name_list = [os.path.basename(x) for x in pathname_list]
    data_dict = { 'img':name_list }
    for i in range(0,10):
        data_dict['c'+str(i)]= all_probs[:,i]

    my_solution = pandas.DataFrame.from_dict(data_dict)

    # Check that your data frame has 418 entries
    print 'csv shape', my_solution.shape
    print my_solution.describe()
    pandas.set_option('display.width', 800)
    print(my_solution)


    # Write your solution to a csv file with the name my_solution.csv
    filename='state_farm_PE_'+filenameNoPrefix+strftime("_%Y-%m-%d_%H-%M", gmtime())+".csv"
    my_solution.to_csv(filename,
                       columns=['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
                       , index=False,index_label = ["img"])
    print 'save complete to file '+filename
    
    #now cut few rows
    my_solution = my_solution[0:79726]
    filename='state_farm_PE_'+filenameNoPrefix+'_cut79726_'+strftime("_%Y-%m-%d_%H-%M", gmtime())+".csv"
    my_solution.to_csv(filename,
                       columns=['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
                       , index=False,index_label = ["img"])
    print 'save complete to file '+filename
 
    for i in range(0,10):
        my_solution['c'+str(i)] = my_solution['c'+str(i)].map('{:,.8f}'.format)
        filename='state_farm_P8_'+filenameNoPrefix+strftime("_%Y-%m-%d_%H-%M", gmtime())+".csv"
    my_solution.to_csv(filename,
                       columns=['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
                       , index=False,index_label = ["img"])
    print 'save complete to file '+filename
    
def calculate_csv(m,classifer_index,load_transform_method, file_name):
    pathname_list,all_probs= predict_directory(m,test_data_dir,
                                                 load_transform_method= load_transform_method,
                                                 transform_folder_to_label=lambda folder: np.ones((1,10)),
                                                 file_name_transform=lambda fullname: fullname,
                                                 classifer_index=classifer_index,
                                                 samples=None)
    predict_to_csv(file_name+'_',pathname_list,all_probs)

    
    
def usage():
   print 'uncomment this'
   # classes_list,all_probs = predict_val_and_score(combined,0,googlenet_load_transform,samples=50)
   # print_report(classes_list,all_probs,labelMapping)
   # calculate_googlenet_csv(combined,0,googlenet_load_transform,'good_file_name')