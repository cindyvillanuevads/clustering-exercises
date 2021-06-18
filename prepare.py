import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler





def unique_cntvalues (df, max_unique):
    '''
    takes in a df and a max_unique values to show using value_counts().
    returns a report of number of unique values foe each columns and shouw unique values that < max_unique
    '''
    #checking the uniques values for each column
    columns = df.columns.tolist()
    print( '************************** COUNT OF UNIQUE VALUES ************************** ')
    print( 'Columns')
    print(" ")
    cat_list = []
    for col in columns:
        print(f'{col} --> {df[col].nunique()} unique values')
        if df[col].nunique() < max_unique:
            cat_list.append(col)
        print(" ")
    #checking  the variables that have few values
    print(" **************************  UNIQUE VALUES **************************")
    print(" ")
    print(f"Uniques values of all the columns that have less than {max_unique} unique values ")
    print(" ")
    for l in cat_list:
        print(l)
        print(df[l].value_counts().sort_index())
        print("--------------------------- ")
        print(" ")





# plot distributions
def distribution (df):
    '''
    takes in a df and plot individual variable distributions excluding object type
    '''
    cols =df.columns.to_list()
    for col in cols:
        if df[col].dtype != 'object':
            plt.hist(df[col])
            plt.title(f'Distribution of {col}')
            plt.xlabel('values')
            plt.ylabel('Counts of customers')
            plt.show()




def distribution_boxplot (df):
    '''
    takes in a df and boxplot variable distributions excluding object type
    '''
    cols =df.columns.to_list()
    for col in cols:
        if df[col].dtype != 'object':
            plt.figure(figsize=(8, 6))
            sns.boxplot(x= col, data=df)
            plt.title(f'Distribution of {col}')
            plt.xlabel('values')
            plt.show()
    return

def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    random_state=123
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')                                  
    return train, validate, test

def split_Xy (train, validate, test, target):
    '''
    This function takes in three dataframe (train, validate, test) and a target  and splits each of the 3 samples
    into a dataframe with independent variables and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    Example:
    X_train, y_train, X_validate, y_validate, X_test, y_test = split_Xy (train, validate, test, 'Fertility' )
    '''
    
    #split train
    X_train = train.drop(columns= [target])
    y_train= train[target]
    #split validate
    X_validate = validate.drop(columns= [target])
    y_validate= validate[target]
    #split validate
    X_test = test.drop(columns= [target])
    y_test= test[target]

    print(f'X_train -> {X_train.shape}               y_train->{y_train.shape}')
    print(f'X_validate -> {X_validate.shape}         y_validate->{y_validate.shape} ')        
    print(f'X_test -> {X_test.shape}                  y_test>{y_test.shape}') 
    return  X_train, y_train, X_validate, y_validate, X_test, y_test



def encoding(df, cols, drop_first=True):
    '''
    Take in df and list of columns
    add encoded columns derived from columns in list to the df
    '''
    for col in cols:
        # get dummy columns
        dummies = pd.get_dummies(df[col], drop_first=drop_first) 
        # add dummy columns to df
        df = pd.concat([df, dummies], axis=1) 
        
    return df

def scaled_df ( train_df , validate_df, test_df, scaler):
    '''
    Take in a 3 df and a type of scaler that you  want to  use. it will scale all columns
    except object type. Fit a scaler only in train and tramnsform in train, validate and test.
    returns  new dfs with the scaled columns.
    scaler : MinMaxScaler() or RobustScaler(), StandardScaler() 
    Example:
    scaled_df( X_train , X_validate , X_test, RobustScaler())
    
    '''
    
    #get all columns except object type
    columns = train_df.select_dtypes(exclude='object').columns.tolist()
    
    # fit our scaler
    scaler.fit(train_df[columns])
    # get our scaled arrays
    train_scaled = scaler.transform(train_df[columns])
    validate_scaled= scaler.transform(validate_df[columns])
    test_scaled= scaler.transform(test_df[columns])

    # convert arrays to dataframes
    train_scaled_df = pd.DataFrame(train_scaled, columns=columns).set_index([train_df.index.values])
    validate_scaled_df = pd.DataFrame(validate_scaled, columns=columns).set_index([validate_df.index.values])
    test_scaled_df = pd.DataFrame(test_scaled, columns=columns).set_index([test_df.index.values])

    #plot
    for col in columns: 
        plt.figure(figsize=(13, 6))
        plt.subplot(121)
        plt.hist(train_df[col], ec='black')
        plt.title('Original')
        plt.xlabel(col)
        plt.ylabel("counts")
        plt.subplot(122)
        plt.hist(train_scaled_df[col],  ec='black')
        plt.title('Scaled')
        plt.xlabel(col)
        plt.ylabel("counts")



    return train_scaled_df, validate_scaled_df, test_scaled_df

def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.75):
    '''
    takes in a df , proportion of columns and rows that we want to keep
    '''
    threshold = int(round(prop_required_columns * len(df.index),0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns),0))
    df = df.dropna(axis=0, thresh=threshold)

     
    return df


def fill_nan(train, validate, test, cols,  strategy = 'most_frequent'):
    
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values = np.nan, strategy= strategy)
    imputer = imputer.fit(train[cols])
    
    train[cols] = imputer.transform(train[cols])
    validate[cols] = imputer.transform(validate[cols])
    test[cols] = imputer.transform(test[cols])
    
    return train, validate, test


def drop_low_missing_values(df, per = 1 ):
    '''
    takes in a df and the percentage that you want to drop of rows. the defautl value is 1%
     remove the rows that has columuns with missing values less than 1%**
    '''
    
    #drop rows with null values < per %
    lis =((100 * df.isnull().sum() / len(df))> 0) &  ((100 * df.isnull().sum() / len(df))< per)
    col_drop = list(lis[lis == True].index)
    df = df.dropna(axis=0, subset = col_drop)
    
    return df


def remove_outliers(df, col_list, k=1.5):
    ''' remove outliers from the numeric columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[f'{col}'].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]
        
    return df