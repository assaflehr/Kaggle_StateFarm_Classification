import random
import os
import numpy as np
import googlenet.transform 

    
def folder_to_label(folder):
    '''from c5 to numpy-array 1x10  like [0,0,0,0,1,0,0,0,0,0]'''
    label = np.zeros((1,10),dtype=np.int)
    label[0][int(folder[1:])]=1
    return label
    
def folder_to_3_labels(folder):
    '''from c5 to [0,0,0,0,1,0,0,0,0,0]'''
    label = np.zeros((1,10),dtype=np.int)
    c_num = int(folder[1:]) #'c9'
    label[0,c_num]=1
    return [label,label,label]
    
def folder_to_number(folder):
    '''from c5 to numpy-array with 1x1 like [5]'''
    label = np.empty((1,1),dtype=np.int)
    label[0][0]=int(folder[1:])
    return label

                                
def generate_data_and_label(root_path,
                            load_transform_method,
                            transform_folder_to_label=folder_to_label,
                            file_name_transform=lambda fullname: os.path.split(fullname)[1],
                            batch_size=32,
                            shuffle=True,
                            verbose=False  ):
    '''
    Customizable generator for files under directory structure like root/c1 root/c2 etc
    It allows you to define your own method of loading a file, so you can use if for images or binary files.s
    load_transform_method - mandatory- to load img and transform it (mean/channel-swap etc), or load numpy-array
    transform_folder_to_label - by default: get the middle folder (root/middle/file.jpg) like 'c9' return label, like 9
    file_name_transform - by default: from full path to file name.
                         Set to None, if you want the generator to not yield it back at all. len(result)=2   '''
    files_with_labels = [] #list of tuples
    for upper_dir in os.listdir(root_path) :
        filename_list = os.listdir(root_path +'/'+ upper_dir)
        for filename in filename_list:
            files_with_labels.append( (root_path +'/'+ upper_dir+'/'+filename, upper_dir))
    if batch_size>len(files_with_labels):
        print 'ValueError'
        raise ValueError('batch_size {} smaller than number of files {}'.format(batch_size,len(files_with_labels)))
    print 'generate_data_and_label total files=',len(files_with_labels)
    if shuffle:
        random.shuffle(files_with_labels)
    index=0
    iteration_count=0
    while 1:  #infinitiy
        if verbose and iteration_count%100==1:
            print ('iteration_count=',iteration_count)
        batch = files_with_labels[index:index+batch_size]
        missing_few = batch_size - len(batch)
        if missing_few>0:
            batch.extend(files_with_labels[0:missing_few])
            index=missing_few
        else:
            index= index+batch_size
        if verbose: print 'batch:',batch
        
        loaded_batch = np.concatenate([load_transform_method(t[0]) for t in batch])
        if verbose: print 'loaded_batch.shape',loaded_batch.shape
        
        #loaded_labels= np.concatenate([transform_folder_to_label(t[1]) for t in batch])
        #if verbose: print 'loaded_labels shape:',loaded_labels.shape,'content:',loaded_labels
        
        # IF OUTPUT LIST > 1
        #  transform_folder_to_label can return a list of elements with different shapes 1x1 like 2 or 1x10 [0,0,1,0..etc ]
     
        # batch each of the resulting lists by itself:  1st of batch (1,4) (1,20)  (1,30)
        #                                               2nd of batch (1,4) (1,20)  (1,30)
        labels_spread= [transform_folder_to_label(pair[1]) for pair in batch]
        #there are two options, list of [label_aux0 , label_aux1, ...] which should concat to [32xlabel_aux0, 32xlabel_aux1]
        #or just one label
        if type(labels_spread[0])==list:
             tuple_of_lists = zip(*labels_spread)  # (list of many (1,4), list of (1,20)
             loaded_labels=[np.concatenate(one_list) for one_list in tuple_of_lists]
        else:
            loaded_labels = np.concatenate([labels_spread])
            loaded_labels = [loaded_labels ] #keras validation generator always return this in a list
        
  #      if verbose: 
  #          print 'loaded_labels list of size ' ,len(loaded_labels),'shape of 1st',loaded_labels[0].shape
        
        if file_name_transform==None:
            yield (loaded_batch, loaded_labels)
        else:
            yield (loaded_batch, loaded_labels,[ file_name_transform(t[0]) for t in batch])

model=None
def predict_directory(model,directory,load_transform_method,transform_folder_to_label,
                                    file_name_transform,classifer_index,samples=None):
    def count_files(directory):
       count=0
       for upper_dir in os.listdir(directory) :
           filename_list = os.listdir(directory +'/'+ upper_dir)
           for filename in filename_list:
               count=count+1
       return count
       
    if (samples==None):
        samples= count_files(directory)
    batch_size=32
    submit_generator= generate_data_and_label(directory,load_transform_method=load_transform_method,
                                                        transform_folder_to_label=transform_folder_to_label,
                                                        file_name_transform=file_name_transform,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        verbose=False)
    num_of_batchs = samples/batch_size
    mod = samples%batch_size
    if mod: 
        num_of_batchs+=1
        
    list_of_probs_batches=[]
    all_file_names=[]
    for batch_i in range(0,num_of_batchs):
        data_batch,_,file_names= next(submit_generator)  #batch of 32 images
        #x = np.asarray(x_list)
        curr_probs = model.predict( data_batch, batch_size)
       
        if classifer_index: #None for only 1, or can be a list from which we will take one classifer
            curr_probs = curr_probs[classifer_index]  
        list_of_probs_batches.append(curr_probs)
        all_file_names.append(file_names)
        if batch_i%5==1:
           print batch_i,  'of size' , batch_size           
    all_probs =  np.concatenate(list_of_probs_batches,axis=0)
    return sum(all_file_names,[]) , all_probs

    

def usage():
    PATH_TO_IMGS = 'C:/code/git/data/statefarm/imgs/'
    
    assert folder_to_label('c1').shape == (1,10)
    batch_generator= generate_data_and_label(PATH_TO_IMGS+'validation',
                                       googlenet.transform.load_transform,
                                       batch_size=3,shuffle=False,verbose=True)         
    data_batch,label_batch,file_names = next(batch_generator)
    print data_batch.shape
    print 'label_batch=' , type(label_batch), 'len',len(label_batch), '[0].shape=' ,label_batch[0].shape
    print file_names
    
    labels_3_generator= generate_data_and_label(PATH_TO_IMGS+'validation',
                                       googlenet.transform.load_transform,
                                       transform_folder_to_label=folder_to_3_labels,
                                       batch_size=3,shuffle=False,verbose=True)         
    data_batch,label_batch,file_names = next(labels_3_generator)
    print '3labels=',label_batch
     
    binary_np_generator=generate_data_and_label(PATH_TO_IMGS+'googlenet/train/out_1' ,
                                       load_transform_method= lambda filename: np.load(open(filename,"rb")),
                                       file_name_transform=None,batch_size=2,shuffle=True,verbose=True) 
    binary_batch,label_batch = next(binary_np_generator)
    
if __name__ == "__main__"    :
    usage()   