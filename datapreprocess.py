import pandas as pd
import numpy as np
from datetime import datetime
import gc
import tool
from sklearn.model_selection import train_test_split

''' 
load data and initial
@parament train_path is the train data's path
@parament test_path is the test data's path
@parament yield_path is the yield data's path
@parament time_path is the time data's path
'''
def data_load(train_path,test_path,yield_path,time_path):
    # train
    df_train = pd.read_csv(train_path)
    df_train['type'] = 'train' # In data process will divide to train an

    # test
    df_test = pd.read_csv(test_path)
    df_test.columns = ['product_pid', 'transaction_date', 'apply_amt', 'redeem_amt','net_in_amt']
    df_test['type'] = 'test'

    # final yield curve info
    df_cbyieldcurve_info_final = pd.read_csv(yield_path)
    df_cbyieldcurve_info_final.columns = ['transaction_date','yield']

    # time table
    df_time_info_final = pd.read_csv(time_path)
    df_time_info_final.rename(columns = {'stat_date':'transaction_date'},inplace = True)

    #-------------------------------Data processing---------------------------------#
    # Merge data
    df_raw = pd.concat([df_train,df_test],ignore_index=True)
    df_raw = df_raw.merge(df_cbyieldcurve_info_final,how = 'left', on = 'transaction_date')
    df_raw = df_raw.merge(df_time_info_final,how = 'left', on = 'transaction_date')

    # Datetime
    df_raw['date'] = pd.to_datetime(df_raw['transaction_date'], format='%Y%m%d')
    df_raw['next_trade_date'] = pd.to_datetime(df_raw['next_trade_date'], format='%Y%m%d')
    df_raw['last_trade_date'] = pd.to_datetime(df_raw['last_trade_date'], format='%Y%m%d')

    df_raw['month'] = df_raw.date.dt.month.astype("int8")
    df_raw['day_of_month'] = df_raw.date.dt.day.astype("int8")
    df_raw['day_of_year'] = df_raw.date.dt.dayofyear.astype("int16")
    df_raw['week_of_month'] = (df_raw.date.apply(lambda d: (d.day-1) // 7 + 1)).astype("int8")
    df_raw['week_of_year'] = (df_raw.date.dt.isocalendar().week).astype("int8")
    df_raw['day_of_week'] = (df_raw.date.dt.dayofweek + 1).astype("int8")
    df_raw['year'] = df_raw.date.dt.year.astype("int32")
    df_raw["is_wknd"] = (df_raw.date.dt.weekday // 4).astype("int8")
    df_raw["quarter"] = df_raw.date.dt.quarter.astype("int8")
    df_raw['is_month_start'] = df_raw.date.dt.is_month_start.astype("int8")
    df_raw['is_quarter_start'] = df_raw.date.dt.is_quarter_start.astype("int8")
    df_raw['is_year_start'] = df_raw.date.dt.is_year_start.astype("int8")
    # 0: Winter - 1: Spring - 2: Summer - 3: Fall
    df_raw["season"] = np.where(df_raw.month.isin([12,1,2]), 0, 1)
    df_raw["season"] = np.where(df_raw.month.isin([6,7,8]), 2, df_raw["season"])
    df_raw["season"] = pd.Series(np.where(df_raw.month.isin([9, 10, 11]), 3, df_raw["season"])).astype("int8")
    df_raw['open_days'] = df_raw.date.apply(lambda x: (x - datetime.strptime(str(20221110), "%Y%m%d")).days)

    # The difference between each transaction date and the first transaction date of the corresponding product
    grouped = df_raw.groupby('product_pid')
    first_transaction_dates = grouped['date'].min()
    df_raw['days_since_first_transaction'] = df_raw.apply(lambda row: (row['date'] - first_transaction_dates[row['product_pid']]).days, axis=1)

    # The date interval between adjacent transactions
    df_raw['days_since_next_trade_date'] = df_raw.apply(lambda row : (row['date'] - row['next_trade_date']).days ,axis = 1)
    df_raw['days_since_last_trade_date'] = df_raw.apply(lambda row : (row['date'] - row['last_trade_date']).days ,axis = 1)

    # Garbage Collection
    del df_train,df_cbyieldcurve_info_final,df_time_info_final
    gc.collect()

    df_raw = df_raw[['product_pid','transaction_date','type','apply_amt', 'redeem_amt', 'net_in_amt','days_since_last_trade_date', 'days_since_next_trade_date', 'day_of_week', 'is_week_end', 
                    'is_wknd', 'open_days', 'days_since_first_transaction']]
    df_raw.to_csv('./raw.csv')
    return df_raw

def data_process(df_raw, val =True):
    # The product_ID in the train cannot be fully matched in the test
    df_train = df_raw[df_raw['type'] == 'train'].reset_index(drop=True)
    df_test = df_raw[df_raw['type'] == 'test'].reset_index(drop=True)
    test_id = tool.CSV(df_test).product_ID

    df_train['type'] = 'val'
    for id in test_id:
        df_train.loc[df_train['product_pid'] == id, 'type'] = 'train'
    df_val = df_train[df_train['type'] == 'val'].reset_index(drop=True)
    df_train = df_train[df_train['type'] == 'train'].reset_index(drop=True)
    if val == True:
        df_train.to_csv('./data_for_val/train.csv')
        df_test.to_csv('./data_for_val/test.csv')
        df_val.to_csv('./data_for_val/val.csv')
    else: 
        df_train.to_csv('./data/train.csv')
        df_test.to_csv('./data/test.csv')
        df_val.to_csv('./data/val.csv')

    return df_train, df_test, df_val


def data_process_no_val(df_raw,val = True):
    # The product_ID in the train cannot be fully matched in the test
    df_train = df_raw[df_raw['type'] == 'train'].reset_index(drop=True)
    df_test = df_raw[df_raw['type'] == 'test'].reset_index(drop=True)

    if val == True:
        df_train.to_csv('./data_for_val/train.csv')
        df_test.to_csv('./data_for_val/test.csv')
        
    else: 
        df_train.to_csv('./data/train.csv')
        df_test.to_csv('./data/test.csv')
        

    return df_train, df_test
    