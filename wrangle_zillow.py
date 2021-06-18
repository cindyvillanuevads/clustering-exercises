import pandas as pd
import numpy as np
import os
import acquire as a


# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.75):
    '''
    takes in a df , proportion of columns and rows that we want to keep
    '''
    threshold = int(round(prop_required_columns * len(df.index),0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns),0))
    df = df.dropna(axis=0, thresh=threshold)
    
    #drop rows with null values < 1
    lis =((100 * df.isnull().sum() / len(df))> 0) &  ((100 * df.isnull().sum() / len(df))< 1)
    col_drop = list(lis[lis == True].index)
    df = df.dropna(axis=0, subset = col_drop)
     
    return df



def wrangle_zillow ( sql_query, prop_required_columns=0.5, prop_required_row=0.75):
    
    #acquire data
    df= a.get_data_from_sql('zillow',sql_query)
    
    #getting the latest transactions 
    df1 = df.sort_values(by ='transactiondate', ascending=True).drop_duplicates( subset = 'parcelid' ,keep= 'last')
    
    #this list has all types of single unit properties
    single= ['Single Family Residential',' Mobile Home' , 'Townhouse '  ]
    #create a mask
    single_mask = df1['propertylandusedesc'].isin(single)
    #using that mask and also add  a condition
    df_single = df1[single_mask & ((df1['unitcnt'] == 1) | (df1['unitcnt'].isnull()))]
    
    #missing values
    df_clean = handle_missing_values(df_single, prop_required_columns, prop_required_row)
    
    #fill missing values in unitcnt
    df_clean['unitcnt'].fillna(1, inplace= True)
    
    #fill missing values heatingorsystemtypeid
    most_f =df_clean['heatingorsystemtypeid'].mode()[0]
    df_clean['heatingorsystemtypeid'].fillna(most_f, inplace= True)
    
    #drop duplicated rows
    df_clean= df_clean.drop(columns = 'heatingorsystemdesc')
    
    return df_clean