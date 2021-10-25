import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string

from src.utils.text import _remove_punctuation
from src.utils.num_vars import find_coe_date


def generate_categories(df: pd.DataFrame) -> list:
  df["category_clean"] = df["category"].apply(lambda x: [_remove_punctuation(i.strip()) for i in x.split(",")])
  all_category = []
  for i in df["category_clean"]:
    all_category += i
  all_category_clean = [i for i in list(set(all_category)) if i != '']
  
  return all_category_clean

def add_category_features(df: pd.DataFrame, category_ls: list) -> pd.DataFrame:
  df["category_clean"] = df["category"].apply(lambda x: [_remove_punctuation(i.strip()) for i in x.split(",")])
  for i in category_ls:
      col_name = f"is_{i.replace(' ', '_')}"
      df[col_name] = df["category_clean"].apply(lambda x: np.where(i in x, 1, 0))
  
  return df

def add_basic_text_features(df: pd.DataFrame) -> pd.DataFrame:
  text_cols = ["features", "accessories", "description"]
  eng_stopwords = stopwords.words('english')
  
  for col in text_cols:
      df[f"{col}_num_words"] = df[col].apply(lambda x: len(str(x).split()))
      df[f"{col}_num_unique_words"] = df[col].apply(lambda x: len(set(str(x).split())))
      df[f"{col}_num_chars"] = df[col].apply(lambda x: len(str(x)))
      df[f"{col}_num_chars"] = df[col].apply(lambda x: len(str(x)))
      df[f"{col}_num_stopwords"] = df[col].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
      df[f"{col}_num_punctuations"] =df[col].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )        
      df[f"{col}_mean_word_len"] = df[col].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
  
  return df

def add_speicific_word_features(df: pd.DataFrame) -> pd.DataFrame:
  for i in ['warranty', 'loan']:
      df[f"is_{i}_features"] = np.where(df["features"].str.contains(i), 1, 0) 
      df[f"is_{i}_description"] = np.where(df["description"].str.contains(i), 1, 0) 
      df[f"is_{i}_accessories"] = np.where(df["accessories"].str.contains(i), 1, 0) 

      df[f"is_{i}"] = np.where((df[f"is_{i}_features"] == 1) |  
                              (df[f"is_{i}_description"] == 1) |
                              (df[f"is_{i}_accessories"] == 1), 1, 0)
  return df

def add_one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
  col_ls = ["transmission", "fuel_type", "type_of_vehicle"]
  
  for col in col_ls:
      unique_vals = list(df[col].unique())
      for val in unique_vals:
          df[f"is_{val}"] = np.where(df[col] == val, 1, 0)
  
  return df

def add_make_model(df: pd.DataFrame) -> pd.DataFrame:
  df['make_fill'] = df.apply(lambda x: x["make"] if not pd.isnull(x["make"]) else x["title"].lower().split()[0], axis = 1)
  df["make_model"] = df["make_fill"] + " " + df["model"]
  return df

def add_time_features(df):
  df["is_lifespan_missing"] = np.where(df["lifespan"].isna(), 1, 0)
  
  df["lifespan"] = pd.to_datetime(df["lifespan"])
  df["reg_date"] = pd.to_datetime(df["reg_date"])
  df["lifespan_year"] = df["lifespan"].dt.year
  df["lifespan_month"] = df["lifespan"].dt.month
  df["reg_date_year"] = df["reg_date"].dt.year
  df["reg_date_month"] = df["reg_date"].dt.month
  
  df["manufactured"] = df.apply(lambda x: np.NaN if x["reg_date_year"] < x["manufactured"] else x["manufactured"], axis=1)
  df["manufactured_to_reg_year"] = df["reg_date_year"] - df["manufactured"]
  df["lifespan_to_reg_year"] = df["lifespan_year"] - df["reg_date_year"]
  df["car_age"] = df.apply(lambda x: np.ceil(((2021-x['reg_date_year']) * 12 + 10-x['reg_date_month'])/12), axis = 1)
  df["car_age_manufactured"] = df.apply(lambda x: 2021-x['manufactured'], axis = 1)
  
  return df

def add_coe_date_features(df: pd.DataFrame) -> pd.DataFrame:
  df['coe_date'] = df.apply(lambda x: find_coe_date(x['title'], x['is_coe_car']), axis = 1)
  df['coe_date_year'] = df['coe_date'].dt.year
  df['coe_date_month'] = df['coe_date'].dt.month 
  df['coe_month_left'] = df.apply(lambda x: (x['coe_date_year']-2021)*12 + x['coe_date_month']-10, axis=1)
  return df