import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures,FunctionTransformer
from Data_Helper import *
import joblib
from update_metadata import update_metadata
import datetime

SEED=42
np.random.seed(SEED)
def log_transform(x): 
        return np.log1p(np.maximum(x, 0))
def with_suffix(_, names: list[str]):  # https://github.com/scikit-learn/scikit-learn/issues/27695
        return [name + '__log' for name in names]

def approach1(x_train,t_train,x_val,t_val,x_test,t_test):
    numeric_feats=['distance',
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    categorial_feats=['pickup_dayofweek','pickup_month','pickup_hour','passenger_count','pickup_day','store_and_fwd_flag']
    Selected_feats=numeric_feats+categorial_feats
    column_transformer=ColumnTransformer([('ohe',OneHotEncoder(handle_unknown="ignore"),categorial_feats),
                                          ('scaling',choose_preprocessing(1),numeric_feats)],remainder='passthrough')
    pipeline=Pipeline(steps=[('ohe',column_transformer),
                          
                             ('regression',Ridge(alpha=1,random_state=SEED))])
  
    model=pipeline.fit(x_train,t_train)
    

    train_r2=predict_eval(model,x_train,t_train,"train")

    val_r2=predict_eval(model,x_val,t_val,"val")
    test_r2=predict_eval(model,x_test,t_test,"test")

    return model,column_transformer,Selected_feats,train_r2,val_r2,test_r2



def approach2(x_train,t_train,x_val,t_val,x_test,t_test):
    numeric_feats=['distance',
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    categorial_feats=['pickup_dayofweek','pickup_month','pickup_hour','passenger_count','pickup_day','store_and_fwd_flag','vendor_id']

    Selected_feats=numeric_feats+categorial_feats



    numeric_transformer = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=3,include_bias=False,interaction_only=False)),
        ('scaler', choose_preprocessing(2))
        
        
    ])

    categorical_transformer = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown="ignore"))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_feats),
            ('categorical', categorical_transformer, categorial_feats)
        ],
        remainder='passthrough'
    )

    x_train=column_transformer.fit_transform(x_train)
    x_val=column_transformer.transform(x_val)
    x_test=column_transformer.transform(x_test)
    pipeline=Pipeline(steps=[
    
                            ('regression',Ridge(alpha=1,random_state=SEED))])
  
    model=pipeline.fit(x_train,t_train)
    

   
    train_r2=predict_eval(model,x_train,t_train,"train")

    val_r2=predict_eval(model,x_val,t_val,"val")
    test_r2=predict_eval(model,x_test,t_test,"test")


    return model,column_transformer,Selected_feats,train_r2,val_r2,test_r2

    ########################


def approach3(x_train,t_train,x_val,t_val,x_test,t_test): #using log
    numeric_feats=['distance',
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    categorial_feats=['pickup_dayofweek','pickup_month','pickup_hour','passenger_count','pickup_day','store_and_fwd_flag','vendor_id']


    
    def log_transform(x): 
        return np.log1p(np.maximum(x, 0))
    def with_suffix(_, names: list[str]):  # https://github.com/scikit-learn/scikit-learn/issues/27695
        return [name + '__log' for name in names]
    LogFeatures = FunctionTransformer(log_transform, feature_names_out=with_suffix)
    numeric_transformer = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=6,include_bias=False,interaction_only=False)),
        ('scaler', choose_preprocessing(2)) ,
        ('log', LogFeatures)
        
    ])

    Selected_feats=numeric_feats+categorial_feats


    categorical_transformer = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown="ignore"))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_feats),
            ('categorical', categorical_transformer, categorial_feats)
        ],
        remainder='passthrough'
    )

    x_train=column_transformer.fit_transform(x_train)
    x_val=column_transformer.transform(x_val)
    x_test=column_transformer.transform(x_test)

    pipeline=Pipeline(steps=[
                        
                            ('regression',Ridge(alpha=1,random_state=SEED))])
  
    model=pipeline.fit(x_train,t_train)
    

    train_r2=predict_eval(model,x_train,t_train,"train")

    val_r2=predict_eval(model,x_val,t_val,"val")
    test_r2=predict_eval(model,x_test,t_test,"test")

    return model,column_transformer,Selected_feats,train_r2,val_r2,test_r2


def approach4(x_train,t_train,x_val,t_val,x_test,t_test): #remove outliers from train_data first 
    df_combined = pd.concat([x_train, t_train], axis=1)
    print(df_combined.shape)
    df_combined=remove_outlires(df_combined,"passenger_count","trip_duration",0)
    t_train_new=df_combined["trip_duration"]
    x_train_new=df_combined.drop(columns=["trip_duration"])

    
    numeric_feats=['distance',
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    categorial_feats=['pickup_dayofweek','pickup_month','pickup_hour','passenger_count','pickup_day','store_and_fwd_flag','vendor_id']

    Selected_feats=numeric_feats+categorial_feats

    
    def log_transform(x): 
        return np.log1p(np.maximum(x, 0))
    def with_suffix(_, names: list[str]):  # https://github.com/scikit-learn/scikit-learn/issues/27695
        return [name + '__log' for name in names]
    LogFeatures = FunctionTransformer(log_transform, feature_names_out=with_suffix)
    numeric_transformer = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=6,include_bias=False,interaction_only=False)),
        ('scaler', choose_preprocessing(2)) ,
        ('log', LogFeatures)
        
    ])



    categorical_transformer = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown="ignore"))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_feats),
            ('categorical', categorical_transformer, categorial_feats)
        ],
        remainder='passthrough'
    )

    x_train_new=column_transformer.fit_transform(x_train_new)
    x_val=column_transformer.transform(x_val)
    x_test=column_transformer.transform(x_test)

    pipeline=Pipeline(steps=[
                        
                            ('regression',Ridge(alpha=1,random_state=SEED))])
  
    model=pipeline.fit(x_train_new,t_train_new)
    

    train_r2=predict_eval(model,x_train_new,t_train_new,"train")

    val_r2=predict_eval(model,x_val,t_val,"val")
    test_r2=predict_eval(model,x_test,t_test,"test")

    return model,column_transformer,Selected_feats,train_r2,val_r2,test_r2





def approach5(x_train,t_train,x_val,t_val):

    numeric_feats=['distance','manhattan_distance',
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude','direction']
    categorial_feats=['pickup_dayofweek','pickup_month','pickup_hour','passenger_count','pickup_day','store_and_fwd_flag','vendor_id']
    Selected_feats=numeric_feats+categorial_feats

    
    df_combined = pd.concat([x_train, t_train], axis=1)
    df_combined=remove_outlires(df_combined,"passenger_count","trip_duration",1)
    print(df_combined.shape)


    t_train_new=df_combined["trip_duration"]
    x_train_new=df_combined.drop(columns=["trip_duration"])


   
    LogFeatures = FunctionTransformer(log_transform, feature_names_out=with_suffix)
    numeric_transformer = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=6,include_bias=False,interaction_only=False)),
        ('scaler', choose_preprocessing(2)) ,
        ('log', LogFeatures)
        
    ])



    categorical_transformer = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown="ignore"))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_feats),
            ('categorical', categorical_transformer, categorial_feats)
        ],
        remainder='passthrough'
    )

    x_train_new=column_transformer.fit_transform(x_train_new)

    x_val=column_transformer.transform(x_val)

    # x_test=column_transformer.transform(x_test)
    pipeline=Pipeline(steps=[
                        
                            ('regression',Ridge(alpha=1,random_state=SEED))])
  
    model=pipeline.fit(x_train_new,t_train_new)

    train_r2=predict_eval(model,x_train_new,t_train_new,"train")

    val_r2=predict_eval(model,x_val,t_val,"val")
    # test_r2=predict_eval(model,x_test,t_test,"test")


    return model,column_transformer,Selected_feats,train_r2,val_r2



    
data_path_train=r"split/train.csv"
data_path_val=r"split/val.csv"
data_path_test=r"split_sample/test.csv"


x_train,t_train=prepare_data(data_path_train)

x_val,t_val=prepare_data(data_path_val)
# x_test,t_test=prepare_data(data_path_test)





# approach1(x_train,t_train,x_val,t_val)  train r2=.429 , val r2=.4441
# approach3(x_train,t_train,x_val,t_val) #poly degree 3 & standard scaler train R2=0.6224286181921248 val R2=0.6428725952492766
# Model,column_transformer,Selected_feats,train_r2,val_r2,test_r2=approach1(x_train,t_train,x_val,t_val)
Model,column_transformer,Selected_feats,train_r2,val_r2=approach5(x_train,t_train,x_val,t_val)
#try your best
# ok approcah 2 is good for now we try remove outlires from specific columns as eda
# do like approach 3 for more systamtic way and 
# read document if you have  a problem
#using log in approach 3 increased performance but not too much so i will try outlires and report
#consider in approcah 4 or 5 using multiple distances and see if one is best or must all
#then consider metadata for this fkn project learn how to do it is ez you have all approaches
#approcah 5 add bearing (direction) and may use delete in ferquent categories

model_data = {
        'version': 'approach5_Model',
        'version_description': "Using approach 5 to Model with all data",
        'num_rows_train': len(x_train),
        'num_rows_val': len(x_val),
        'model':Model,
        'data_path_train':data_path_train,
        'data_path_val':data_path_val,
        'data_path_test':data_path_test,
        'data_preprocessor':column_transformer,
        'random_Seed':SEED,
        'Selected_Features':Selected_feats,
        'train_r2':train_r2,
        'val_r2':val_r2,
        # 'test_r2':test_r2,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
update_metadata(model_data,1)
now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
filename = f'Model_ {now}_R2_train {train_r2:.2f} R2_val {val_r2:.2f}.pkl'
joblib.dump(model_data, filename)

print(f"Model saved as {filename}")
