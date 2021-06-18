import pandas as pd
import numpy as np
import os


# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")


# *************************************  connection url **********************************************

# Create helper function to get the necessary connection url.
def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    from env import host, username, password
    return f'mysql+pymysql://{username}:{password}@{host}/{db_name}'

# ************************************ generic acquire function ***************************************************************


def get_data_from_sql(db_name, query):
    """
    This function takes in a string for the name of the database that I want to connect to
    and a query to obtain my data from the Codeup server and return a DataFrame.
    db_name : df name in a string type
    query: aalready created query that was named as query 
    Example:
    query = '''
    SELECT * 
    FROM table_name;
    '''
    df = get_data_from_sql('zillow', query)
    """
    df = pd.read_sql(query, get_connection(db_name))
    return df

# ************************************ unique values ***************************************************************

def report_unique_val (df):
    '''
    takes in a df and gives you a report of number of unique values and count values <15 (categorical)
    count values <15 (numerical)
    '''
    num_cols = df.select_dtypes(exclude = 'O').columns.to_list()
    cat_cols = df.select_dtypes(include = 'O').columns.to_list()
    for col in df.columns:
            print(f'**{col}**')
            le = df[col].nunique()
            print ('Unique Values : ', df[col].nunique())
            print(' ')
            if col in cat_cols and le < 15:
                print(df[col].value_counts())
            if col in num_cols and  le < 23:
                 print(df[col].value_counts().sort_index(ascending=True)) 
            elif col in num_cols and le <150:
                print(df[col].value_counts(bins=10, sort=False).sort_index(ascending=True))
            elif col in num_cols and le <1001:
                print(df[col].value_counts(bins=100, sort=False).sort_index(ascending=True))

            print('=====================================================')

def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # shape
    # .head()
    # .info()
    # .describe()
    # value_counts()
    # observation of nulls in the dataframe
    '''
    print('=====================================================')
    print('Dataframe shape: ')
    print(df.shape)
    print('=====================================================')
    print('Dataframe head: ')
    print(df.head(3))
    print('=====================================================')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================')
    print('Dataframe Description: ')
    print(df.describe().T)
    print('=====================================================')
    report_unique_val (df)

def nulls_by_row(df):
    '''
    akes in a dataframe and returns a dataframe with 3 columns: 
    the number of columns missing, percent of columns missing, and number of rows with n columns missing
    '''
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing




def miss_dup_values(df):
    '''
    takes in a dataframe of observations and attributes and returns a dataframe where each row is an atttribute name, 
    the first column is the number of rows with missing values for that attribute, 
    and the second column is percent of total rows that have missing values for that attribute and 
    duplicated rows.
    '''
        # Total missing values
    mis_val = df.isnull().sum()
        # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        #total of duplicated
    dup = df.duplicated().sum()  
        # Percentage of missing values
    dup_percent = 100 * dup / len(df)
        # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
           "There are " + str(mis_val_table_ren_columns.shape[0]) +
           " columns that have missing values.")
    print( "  ")
    print (f"** There are {dup} duplicate rows that represents {round(dup_percent, 2)}% of total Values**")
        # Return the dataframe with missing information
    return mis_val_table_ren_columns


def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.75):
    '''
    takes in  a df , aproportion of columns and rows that we want to keep and 
    drop rows or columns based on the percent of values that are missing:
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
