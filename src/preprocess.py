import pandas as pd
import numpy as np
from datetime import datetime
def get_days(datetime):
    datetime.days
def preprocess(df):
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] =  pd.to_datetime(df['signup_date'])
    cut_off_date = datetime.strptime('2014-07-01', '%Y-%m-%d')
    churn_date = datetime.strptime('2014-06-01', '%Y-%m-%d')
    df['churn'] = (df['last_trip_date'] < churn_date).astype(int)
    # df['churn'] = df['churn'].apply(lambda x: x.day)
    df['days_until_cutoff'] = (cut_off_date - df['signup_date'])
    df['days_until_cutoff'].apply(get_days)
    df['days_until_cutoff'] = (df['days_until_cutoff'] / np.timedelta64(1, 'D')).astype(int)
    df.replace({False:0, True:1}, inplace=True)
    df = df.dropna(axis=0, how='any')
    df = pd.get_dummies(df)
    df.pop('last_trip_date')
    df.pop('signup_date')
    return df
