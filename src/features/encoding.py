import pandas as pd
import numpy as np
import pandas as pd

from src.utils.agg import percentile
from src.features.transformations import add_make_model
from src.features.imputation import backfill_missing_cat_var

def col_encoding(df, input_var, output_var):
  df_agg = df.groupby(input_var).agg({output_var: [np.mean,
                                                  np.min, 
                                                  np.max,
                                                  "count",
                                                  np.std,
                                                  np.median,
                                                  percentile(25),
                                                  percentile(50),
                                                  percentile(75)]}).reset_index()
  df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
  df_agg.rename(columns = {col: col+f'_{input_var}' for col in df_agg.columns.values if col != f"{input_var}_"},
                inplace = True)
  col_ls = [col for col in df_agg if col != f"{input_var}_"]
  return df_agg, col_ls

def cat_encoding(df, df_train):
  cat_encode_vars = ['make_fill',
                    "make_model",
                    'type_of_vehicle',
                    'transmission',
                    'fuel_type']
  target = "price"
  df_train = add_make_model(df_train)
  df_train = backfill_missing_cat_var(df_train)
  for i in cat_encode_vars:
    df_agg, col_ls = col_encoding(df_train, i, target)
    df = df.merge(df_agg, left_on=i, right_on=f"{i}_", how="left")
    df = df.drop(f"{i}_", axis=1)
    df[col_ls] = df[col_ls].fillna(value=-1)

  return df
      