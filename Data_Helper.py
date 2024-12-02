import pandas as pd
import pandas as pd 
import numpy as np
from geopy  import distance,Point
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from scipy import stats
from geopy.point import Point
import math
from sklearn.metrics import r2_score
import joblib


##### All FNs You Would Need  ^_^  #########

def predict_eval(model,train,target,name):
    y_train_pred=model.predict(train)
    r2=r2_score(target,y_train_pred)
    print(f"{name} R2={r2}")
    return r2

def distance_(df):
    start = (df['pickup_latitude'], df['pickup_longitude'])
    end = (df['dropoff_latitude'], df['dropoff_longitude'])
    dist = distance.geodesic(start, end).km
    return dist


def manhattan_distance(row):
   
    lat_distance = abs(row['pickup_latitude'] - row['dropoff_latitude']) * 111  # approx 111 km per degree latitude
    lon_distance = abs(row['pickup_longitude'] - row['dropoff_longitude']) * 111 * math.cos(math.radians(row['pickup_latitude']))  # adjust for latitude
    
    return lat_distance + lon_distance


def remove_outlires(df,feat_column,target_column,use_speed=0):
    if(use_speed==0):
        factor=4.5
        print("factor used without speed ",factor)

        #from our eda we need to remove outlier from passenger count and trip duration
        z_scores = np.abs(stats.zscore(df[[feat_column,target_column]]))
        filtered_entries = (z_scores < factor).all(axis=1)

        return df[filtered_entries]
    else:
        df["speed"]=df["distance"]/df["trip_duration"]
        factor=4.5
        print("factor used with speed ",factor)
        z_scores = np.abs(stats.zscore(df[[feat_column,target_column,'speed']]))
        filtered_entries = (z_scores < factor).all(axis=1)
        df.drop(columns=["speed"],inplace=True)
    
        df.reset_index(drop=True, inplace=True)
        return df[filtered_entries]
    
def calculate_direction(row):#Bearing
    pickup_coordinates =  Point(row['pickup_latitude'], row['pickup_longitude'])
    dropoff_coordinates = Point(row['dropoff_latitude'], row['dropoff_longitude'])
    
    # Calculate the difference in longitudes
    delta_longitude = dropoff_coordinates[1] - pickup_coordinates[1]
    
    # Calculate the bearing (direction) using trigonometry
    y = math.sin(math.radians(delta_longitude)) * math.cos(math.radians(dropoff_coordinates[0]))
    x = math.cos(math.radians(pickup_coordinates[0])) * math.sin(math.radians(dropoff_coordinates[0])) - \
        math.sin(math.radians(pickup_coordinates[0])) * math.cos(math.radians(dropoff_coordinates[0])) * \
        math.cos(math.radians(delta_longitude))
    
    # Calculate the bearing in degrees
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    
    # Adjust the bearing to be in the range [0, 360)
    bearing = (bearing + 360) % 360
    
    return bearing

def prepare_data(data_path):
    
    df=pd.read_csv(data_path)
    df=df.iloc[:300000]
    #####
    df['trip_duration'] = np.log1p(df['trip_duration'])
    #####
    df['distance']=df.apply(distance_,axis=1)
    df['direction'] = df.apply(calculate_direction, axis=1)
    df['manhattan_distance']=df.apply(manhattan_distance,axis=1)
    ####
    df["pickup_datetime"]=pd.to_datetime( df["pickup_datetime"])
    
    df["pickup_day"]=df["pickup_datetime"].dt.day
    df["pickup_month"]=df["pickup_datetime"].dt.month
    df["pickup_hour"]=df["pickup_datetime"].dt.hour
    df["pickup_dayofweek"]=df["pickup_datetime"].dt.dayofweek


    df.drop(columns=["id","pickup_datetime"],inplace=True)
    
    df.reset_index(drop=True, inplace=True)

    df_t=df["trip_duration"]
    df.drop(columns=["trip_duration"],inplace=True)
    
    df_x=df

    print(f"from Data Helper.py ^_^ , data sahpe: {df_x.shape},target shpape: {df_t.shape} ")

    
    return df_x,df_t



def choose_preprocessing(val=0):
    if(val==1):
        return MinMaxScaler()
    if(val==2):
        return StandardScaler()
    return None
    

def load_model(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except joblib.externals.loky.process_executor.TerminatedWorkerError:
        print("Error: The file could not be loaded.")
    except Exception as e:
        print(f"An error occurred: {e}")
