# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:50:38 2016

@author: Assaf
"""

def saveToFiles(model,filenameWithoutSuffix):
    ''' will create X.json and X.h5 files'''
    with open(filenameWithoutSuffix +'.json', "w") as json_file:
        json_file.write(model.to_json() )
    model.save_weights(filenameWithoutSuffix +'.h5')
    print("Saved model to disk "+ filenameWithoutSuffix)

def loadFromFiles(filenameWithoutSuffix):
    from keras.models import model_from_json
    json_file = open(filenameWithoutSuffix+ '.json', 'r')
    json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(json)
    loaded_model.load_weights(filenameWithoutSuffix +'.h5')
    return loaded_model
    