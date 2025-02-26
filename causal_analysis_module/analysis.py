#%%
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pandas as pd
from pathlib import Path
import pandas_flavor as pf

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from ydata_profiling import ProfileReport

# Get current directory and construct file path
dir_path = os.getcwd()
file_path = os.path.join(dir_path, "00_Raw_Data", "*.csv")
files_path = glob.glob(file_path)

#%%
@pf.register_dataframe_method
def read_csv_files():
    """AI is creating summary for read_csv_files

    Args:
        00_Raw_Data ([.csv]): The path to the raw data

    Returns:
        [pandas.dataframe]: panadas DataFrames used to perform further analyisis and pre-porcessing.
    """

    dir_path = os.getcwd()
    file_path = os.path.join(dir_path, "00_Raw_Data", "*.csv")
    files_path = glob.glob(file_path)

    dataframes = {}
    for file in files_path:
        filename = Path(file).stem
        try:
            df = pd.read_csv(file)
            dataframes[filename] = df
            print(f"Successfully loaded: {filename} with shape {df.shape}")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    
    # Print available dataframes
    print("\nAvailable dataframes:")
    for key in dataframes.keys():
        print(f"- {key}")
    
    return dataframes




@pf.register_dataframe_method
def data_preprocessing():
    
    dataframes = read_csv_files()

    chefmozparking_df = dataframes['chefmozparking']
    chefmozaccepts_df = dataframes['chefmozaccepts']
    userpayment_df = dataframes['userpayment']
    restaurant_geoplaces2_df = dataframes['geoplaces2']
    user_rating_final = dataframes['rating_final']
    usercuisine_df = dataframes['usercuisine']
    chefmozcuisine_df = dataframes['chefmozcuisine']
    chefmozhours4_df = dataframes['chefmozhours4']
    userprofile_df = dataframes['userprofile']

    patrons_df = pd.merge(userprofile_df,userpayment_df , on='userID', how='left')
    patrons_df = patrons_df \
        .merge(usercuisine_df, on='userID', how='left') \
        .merge(user_rating_final, on='userID', how='left')
    
    patrons_df.rename(columns={'latitude':'p.latitude', 'longitude':'p.longitude' }, inplace=True)
    
    restaurant_df = pd.merge(restaurant_geoplaces2_df, chefmozaccepts_df, on='placeID', how='left') 

    restaurant_df = restaurant_df\
        .merge(chefmozparking_df, on='placeID', how='left') \
        .merge(chefmozcuisine_df, on='placeID', how='left') \
        .merge(chefmozhours4_df, on='placeID', how='left') 
    
    
    data = patrons_df\
        .merge(restaurant_df, on='placeID', how='left')
    
    df = data

    # Separate categorical and numerical for further processing
    category_columns = df.select_dtypes(include = ['object']).columns.tolist()
    cat_data_df = df[category_columns]
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_data_df = df[numerical_columns]

    # preprocessing Categorical Data
    cat_data_df.loc[:, 'smoker'] = cat_data_df['smoker'].replace('?', np.nan)
    cat_data_df.loc[:, 'days']= cat_data_df['days'].replace('Mon;Tue;Wed;Thu;Fri;', 'weekdays')
    cat_data_df['dress_preference']= cat_data_df['dress_preference'].replace('?', np.nan)
    cat_data_df['ambience']= cat_data_df['ambience'].replace('?', np.nan)
    cat_data_df.rename(columns={'Rcuisine_x':'User_cuisine', 'name':'restaurant_name', 'Rpayment': 'restaurant_payment', 'Rcuisine_y': 'restaurant_specialty'}, inplace=True)

    #Removing unwanted columns
    cat_data_df.drop(columns = ['fax'])

    # Handling Time
    cat_data_df['hours'] = cat_data_df['hours'].astype(str)

    # Remove the trailing semicolon and split into start and end times
    cat_data_df['start_time'] = cat_data_df['hours'].str[:-1].str.split('-').str[0]
    cat_data_df['end_time'] = cat_data_df['hours'].str[:-1].str.split('-').str[1]

    #convert to datetime
    cat_data_df['start_time'].replace('na', np.nan, inplace=True)
    cat_data_df['end_time'].replace('na', np.nan, inplace=True)

    cat_data_df['start_time'] = pd.to_datetime (cat_data_df['start_time'], format = '%H:%M')
    cat_data_df['end_time'] = pd.to_datetime (cat_data_df['end_time'], format = '%H:%M')

    # cat_data_df
    #return data

    ##preprocessing 
    def categorize_time(row):
        if pd.isna(row['start_time']) or pd.isna(row['end_time']):
            return 'Invalid'

        start_time = row['start_time']
        end_time = row['end_time']

        if start_time >= pd.to_datetime('06:00').time() and end_time <= pd.to_datetime('12:00').time():
            return 'Morning'
        elif start_time >= pd.to_datetime('12:00').time() and start_time < pd.to_datetime('18:00').time():
            return 'Afternoon'
        elif start_time >= pd.to_datetime('18:00').time() or (start_time < pd.to_datetime('06:00').time() and end_time >= pd.to_datetime('06:00').time()): # Evening or Night into next day
            return 'Evening'
        elif end_time < start_time:  # 24-hour: end time is before start time
            return '24H'
        elif start_time >= pd.to_datetime('06:00').time() and end_time >= pd.to_datetime('18:00').time():  # Full Day
            return 'Full Day'
        else:
            return 'Night'  # Covers cases not explicitly handled above
        
    def process_business_hours(df):
        df.replace('na', np.nan, inplace=True)
        df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M').dt.time
        df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M').dt.time
        df['Business_hours'] = df.apply(categorize_time, axis=1)
        return df
    
    cat_data_df = process_business_hours(cat_data_df)
    #return cat_data_df
       
    # address Missing values - Categorical
    columns_with_missing_values_cat = [column for column in cat_data_df.columns if cat_data_df[column].isnull().any()]

    # address Missing values - Numerical 
    columns_with_missing_values_num = [column for column in numerical_data_df.columns if numerical_data_df[column].isnull().any()]
    
    ##Initialize SimpleImputer with the desired strategy
    imputer = SimpleImputer(strategy="most_frequent")

    # Apply imputer to columns with missing values
    cat_data_df[columns_with_missing_values_cat] = imputer.fit_transform(cat_data_df[columns_with_missing_values_cat]) 
    
    final_data_df = pd.concat([numerical_data_df, cat_data_df], axis=1)

    
    return final_data_df


data_preprocessing()



#%%



# %%
