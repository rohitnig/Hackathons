# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 18:10:55 2021

@author: rpnigam
"""

import cloth_tagger as ct
import pandas as pd

# Simple class to save the configuration of the files used to train the model and for predctions

class cloth_config:
    def __init__(self, conf_file = 'config.ini'):
        config = pd.read_csv(conf_file)
        self.train_file = config.head(1).trainfile.values[0]
        self.ip_file = config.head(1).targetfile.values[0]
        self.op_file = config.head(1).opfile.values[0]
        
    def get_files(self):
        return (self.train_file, self.ip_file, self.op_file)
    
if __name__ == '__main__':
    clothsTagger = ct.cloth_tagger()
    cc = cloth_config('config.ini')
    train_file, ip_file, op_file = cc.get_files()
    
    clothsTagger.fit(train_file) ## We can also pass a file to save the model with {model_file='model.sav'}
    predictions = clothsTagger.predict(ip_file) ## We can also pass a file to save the model with {model_file='model.sav'}
    
    predictions.to_csv(op_file, index=False)