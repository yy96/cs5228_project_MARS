import pandas as pd
import numpy as np
import yaml
import os

from src.utils.num_vars import (
  calculate_arf,
  calculate_depreciation,
  calculate_dereg_value,
  calculate_road_tax
)
from src.features.transformations import add_make_model

CURR_DIR = os.path.abspath(os.path.dirname(__file__))
with open(f'{CURR_DIR}/../../config.yaml', 'r') as file:
    feature_dic = yaml.safe_load(file)
target = feature_dic["target"]
cat_encoding_features = feature_dic["cat_encoding_features"]
num_features = feature_dic["num_features"]
cat_features = feature_dic["cat_features_engineer"]

def backfill_arf(df):
  df["arf_calculated"] = df["omv"].apply(lambda x: calculate_arf(x))
  df['arf_fill'] = np.where(df['arf'].isnull(), df['arf_calculated'], df['arf'])
  
  return df

def backfill_dereg_value(df):
  df["dereg_value_calculated"] = df.apply(lambda x: calculate_dereg_value(x['arf_fill'],
                                                                          x['car_age'],
                                                                          x['coe_month_left'], 
                                                                          x['coe_date'], 
                                                                          x['coe_fill'], 
                                                                          x['is_coe_car']), axis = 1)
  df['dereg_value_fill'] = np.where(df['dereg_value'].isnull(), df['dereg_value_calculated'], df['dereg_value'])
  
  return df

def backfill_depreciation(df):
  df["depreciation_calculated"] = df.apply(lambda x: calculate_depreciation(x['omv'], 
                                                                            x['arf_fill'], 
                                                                            x['coe_fill'], 
                                                                            x['dereg_value_fill'], 
                                                                            x['car_age']), axis = 1)
  df['depreciation_fill'] = np.where(df['depreciation'].isnull(), df['depreciation_calculated'], df['depreciation'])
  
  return df

def backfill_road_tax(df):
  df["road_tax_calculated"] = df.apply(lambda x: calculate_road_tax(x['engine_cap'], x['car_age']), axis = 1)
  df['road_tax_fill'] = np.where(df['road_tax'].isnull(), df['road_tax_calculated'], df['road_tax'])
  
  return df

def _look_up_missing(df_origin, var_name, df_lookup, merge_col):
    selected_col = merge_col + [f"{var_name}_mean"]
    df_origin = pd.merge(df_origin, df_lookup[selected_col], how='left', on=merge_col)
    df_origin[f"{var_name}_missing"] = np.where(df_origin[var_name].isnull(), 1, 0)
    df_origin[f"{var_name}_fill"] = np.where(df_origin[var_name].isnull(),
                                           df_origin[f"{var_name}_mean"],
                                           df_origin[var_name])
    return df_origin

def backfill_coe(df, df_train):
  df_train["reg_date"] = pd.to_datetime(df_train["reg_date"])
  df_train["reg_date_year"] = df_train["reg_date"].dt.year
  df_train["reg_date_month"] = df_train["reg_date"].dt.month
  df_coe_lookup = df_train.groupby(['reg_date_year', 'reg_date_month']).agg({'coe':'mean'}).reset_index()
  df_coe_lookup.rename(columns={'coe': 'coe_mean'}, inplace=True) 

  df = _look_up_missing(df, 'coe', df_coe_lookup, ['reg_date_year', 'reg_date_month'])

  return df

def backfill_make_model_features(df, df_train):
  make_model_features = ["curb_weight", "power", "engine_cap"]
  agg_dic = {}
  for i in make_model_features:
      agg_dic[i] = "mean"

  df_train = add_make_model(df_train)
  df_num_vars_lookup = df_train.groupby(["transmission", "make_model", "manufactured"]).agg(agg_dic).reset_index()
  df_num_vars_lookup.rename(columns= {'curb_weight': 'curb_weight_mean', 
                                      'power': 'power_mean', 
                                      'engine_cap': 'engine_cap_mean'}, inplace=True)
  
  df = _look_up_missing(df, "power", df_num_vars_lookup, ["transmission", "make_model", "manufactured"])
  df = _look_up_missing(df, "curb_weight", df_num_vars_lookup, ["transmission", "make_model", "manufactured"])
  df = _look_up_missing(df, "engine_cap", df_num_vars_lookup, ["transmission", "make_model", "manufactured"])
  
  return df

def backfill_mileage(df, df_train):
  df_train["reg_date"] = pd.to_datetime(df_train["reg_date"])
  df_train["reg_date_year"] = df_train["reg_date"].dt.year
  df_train["reg_date_month"] = df_train["reg_date"].dt.month
  df_train["car_age"] = df_train.apply(lambda x: np.ceil(((2021-x['reg_date_year']) * 12 + 10-x['reg_date_month'])/12), axis = 1)
  
  df_mileage_lookup = df_train.groupby('car_age').agg({"mileage": "mean"}).reset_index()
  df_mileage_lookup.rename(columns = {"mileage": "mileage_mean"}, inplace=True)

  df = _look_up_missing(df, "mileage", df_mileage_lookup, ["car_age"])

  return df

def backfill_fuel_type(df, df_train):
  df_train = add_make_model(df_train)
  df_non_missing = df_train[~(df_train['fuel_type'].isna())][['make_model', 'fuel_type']]
  grouped_series = df_non_missing.groupby(['make_model', 'fuel_type']).fuel_type.count()
  grouped_top = pd.DataFrame(grouped_series.groupby(level='make_model').nlargest(1).reset_index(level=0, drop=True))
  grouped_top.rename({"fuel_type": "count"}, axis=1, inplace=True)
  grouped_top.reset_index(inplace=True)
  grouped_top.rename({"fuel_type": "fuel_type_fill"}, axis=1, inplace=True)

  df = df.merge(grouped_top, on='make_model', how ='left')
  df["fuel_type"] = np.where(df['fuel_type'].isnull(), df['fuel_type_fill'], df['fuel_type'])

  return df

def backfill_missing_num_var(df):
  df[num_features] = df[num_features].fillna(value=-1)
  return df

def backfill_missing_cat_var(df):
  df[cat_features] = df[cat_features].fillna(value="missing")
  return df
