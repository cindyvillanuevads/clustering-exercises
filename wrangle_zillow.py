import pandas as pd
import numpy as np
import os
import acquire as a
import prepare as p


# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")



sql_query ='''
SELECT prop.parcelid,  prop.basementsqft, bathroomcnt, bedroomcnt, decktypeid, calculatedfinishedsquarefeet,
fips, fireplacecnt, garagecarcnt, hashottuborspa, latitude, longitude, lotsizesquarefeet, poolcnt,
yearbuilt, numberofstories, prop.airconditioningtypeid, airconditioningdesc, prop.architecturalstyletypeid,
architecturalstyledesc, prop.buildingclasstypeid, buildingclassdesc, prop.heatingorsystemtypeid,
heatingorsystemdesc, prop.storytypeid, storydesc, prop.propertylandusetypeid, propertylandusedesc, 
prop.typeconstructiontypeid, typeconstructiondesc, unitcnt, taxvaluedollarcnt, taxamount, logerror, transactiondate 
from properties_2017 as prop
right join predictions_2017 as pred USING (parcelid)
LEFT JOIN airconditioningtype USING (airconditioningtypeid)
LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
LEFT JOIN buildingclasstype USING (buildingclasstypeid)
LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
LEFT JOIN propertylandusetype USING (propertylandusetypeid)
LEFT JOIN storytype USING(storytypeid)
LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
WHERE transactiondate like '2017%'  
AND latitude != 'NULL' AND longitude != 'NULL';
'''


def wrangle_zillow (  prop_required_columns=0.5, prop_required_row=0.75, strategy = 'most_frequent'):
    '''
    Acquire zillow db from sql, single unit properties with transactions 2017,
    remove outliers, missing values,  split into train validate and test.
    impute heatingorsystemtypeid with most_frequent value
    return tran, validate test

    '''
    
    #acquire data
    df= a.get_data_from_sql('zillow',sql_query)
    
    #getting the latest transactions 
    df1 = df.sort_values(by ='transactiondate', ascending=True).drop_duplicates( subset = 'parcelid' ,keep= 'last')
    
    #this list has all types of single unit properties
    single= ['Single Family Residential',' Mobile Home' , 'Townhouse ', 'Manufactured, Modular, Prefabricated Homes'  ]
    #create a mask
    single_mask = df1['propertylandusedesc'].isin(single)
    #using that mask and also add  a condition
    df_single = df1[single_mask & ((df1['unitcnt'] == 1) | (df1['unitcnt'].isnull()))]
    
    #remove outliers 
    col_list =['calculatedfinishedsquarefeet', 'bedroomcnt', 'bathroomcnt']
    df_single = p.remove_outliers(df_single, col_list, k=1.5)

    #missing values
    df_clean = p.handle_missing_values(df_single, prop_required_columns, prop_required_row)
    
    #missing low values
    df_clean = p.drop_low_missing_values(df_clean)
    
    #drop duplicated rows
    df_clean= df_clean.drop(columns = 'heatingorsystemdesc')
    
    #fill missing values in unitcnt
    df_clean['unitcnt'].fillna(1, inplace= True)
    print('df -->', df_clean.shape)
    #split
    train, validate,  test = p.split_data(df_clean)
    
    #fill nan using simple imputer
    cols = ['heatingorsystemtypeid']
    train, validate, test = p.fill_nan(train, validate, test, cols,  strategy = 'most_frequent')
    
    return train, validate, test