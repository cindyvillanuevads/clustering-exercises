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

    #remove outliers
    col_list = ['age', 'annual_income', 'spending_score']
    df = p.remove_outliers(df, col_list, k=1.5)
    print('df shape after removing outliers', df.shape)
    
    #split data (use my function that is prepare.py)
    train, validate, test =p.split_data(df)
    
    #encode data
    #train
    train = p.encoding(train, ['gender'])
    #validate
    validate = p.encoding(validate, ['gender'])
    #test
    test = p.encoding(test, ['gender'])
    
    #scaling (use my function that is prepare.py)
    train_scaled , validate_scaled , test_scaled = p.scaled_df( train , validate , test,  MinMaxScaler())
    
    return train , validate , test, train_scaled , validate_scaled , test_scaled
