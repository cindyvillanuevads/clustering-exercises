import pandas as pd
import numpy as np
import os
import acquire as a
import prepare as p
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")


def get_mallcustomer_data():
    df = pd.read_sql('SELECT * FROM customers;', a.get_connection('mall_customers'))
    return df.set_index('customer_id')




def wrangle_mall(scaler = MinMaxScaler()):
    '''
    takes in a type of scaler (MinMaxScaler, RobustScaler, StandardScaler)  acquires data from mall_customers.customers in mysql database,
     splits the data into train, validate, and split, creates dummies, and scales using MinMaxScaler
    return train , validate , test
    '''

    #acquire data
    df= get_mallcustomer_data()
    print('df shape', df.shape)
    
    #split data (use my function that is prepare.py)
    train, validate, test =p.split_data(df)
    
    #encode data
    #train
    dummy_train = pd.get_dummies(train[['gender']], dummy_na=False, drop_first=[True])
    train= pd.concat([train, dummy_train], axis=1)
    #validate
    dummy_val = pd.get_dummies(validate[['gender']], dummy_na=False, drop_first=[True])
    validate= pd.concat([validate, dummy_val], axis=1)
    #test
    dummy_test = pd.get_dummies(test[['gender']], dummy_na=False, drop_first=[True])
    test= pd.concat([test, dummy_test], axis=1)
    
    #scaling (use my function that is prepare.py)
    train , validate , test = p.scaled_df( train , validate , test,  scaler)
    
    return train , validate , test
