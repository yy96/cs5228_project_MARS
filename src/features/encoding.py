import pandas as pd
import numpy as np
import pandas as pd

from src.utils.agg import percentile

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
  return df_agg

def cat_encoding(df, df_train):
  cat_encode_vars = ['make_fill',
                    'model',
                    "make_model",
                    'type_of_vehicle',
                    'transmission',
                    'fuel_type']
  target = "price"
  for i in cat_encode_vars:
    df_agg = col_encoding(df_train, i, target)
    df = df.merge(df_agg, left_on=i, right_on=f"{i}_", how="left")
    df = df.drop(f"{i}_", axis=1)

  return df
      