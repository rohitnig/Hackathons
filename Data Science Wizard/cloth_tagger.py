# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:33:11 2021

@author: rpnigam
"""

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from category_encoders.cat_boost import CatBoostEncoder as cate

class cloth_tagger:
    
    def __init__(self):
        self.model = None 
        self.transformer = None
    
    def featureEngg(self, df, y=None):
        df['Unisex'] = 1
        df.loc[df.category.isin(['Blouse', 'Tunic']), 'Unisex'] = 0 # Blouse and Tunic are usually for women
        
        df['Formal'] = 0
        df.loc[df.color.isin(['Black', 'Blue', 'Brown', 'White']), 'Formal'] = 1
        
        if self.transformer is None:
            self.transformer = cate().fit(X=df[['category', 'main_promotion', 'color']], y=y)
        
        encoded = self.transformer.transform(df[['category', 'main_promotion', 'color']])

        ret_df = pd.concat([encoded, df.stars, df.Unisex, df.Formal], axis=1)
        return ret_df
        
    def fit(self, trainFile, model_type = RandomForestClassifier(), model_file = None):
        train = pd.read_csv(trainFile)
        
        y = train.success_indicator.apply(lambda x: 1 if x=='top' else 0)
        X = self.featureEngg(train, y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state=42)
        
        self.model = model_type.fit(X_train, y_train)
        if model_file is not None:
            pickle.dump(self.model, open(model_file, 'wb'))
            pickle.dump(self.transformer, open(model_file+'.enc', 'wb'))
    
    def predict(self, testFile, model = None, model_file = None):
        test = pd.read_csv(testFile)
        
        if self.model is None:
            if model_file is None:
                return None
            else:
                self.model = pickle.load(open(model_file, 'rb'))
                self.transformer = pickle.load(open(model_file+'.enc', 'rb'))

        X = self.featureEngg(test)
        y_predicted = self.model.predict(X)
        y_predicted = pd.DataFrame(y_predicted, columns=['predict_boolean'])
        y_predicted['success_indicator'] = y_predicted.predict_boolean.apply(lambda x: 'flop' if x==0 else 'top')

        return y_predicted[['success_indicator']]

