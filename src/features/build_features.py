import os
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import KFold
from src.features.transformations import (
  generate_categories,
  add_category_features,
  add_basic_text_features,
  add_speicific_word_features,
  add_one_hot_encode,
  add_make_model,
  add_time_features,
  add_coe_date_features)

from src.features.imputation import (
  backfill_arf,
  backfill_dereg_value,
  backfill_depreciation,
  backfill_road_tax,
  backfill_coe,
  backfill_make_model_features,
  backfill_mileage,
  backfill_missing_cat_var,
  backfill_missing_num_var
)

from src.features.encoding import (
  cat_encoding
)

pd.options.mode.chained_assignment = None 
PROJECT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
train_dataset_path = os.path.join(PROJECT_DIR, "data/raw/train.csv")
test_dataset_path = os.path.join(PROJECT_DIR, "data/raw/test.csv")
save_data_path = os.path.join(PROJECT_DIR, "data/processed")
category_ls = generate_categories(pd.read_csv(train_dataset_path))


def main():
  argparser = ArgumentParser()
  argparser.add_argument('--stage', type=str, required=True)
  args = argparser.parse_args()

  stage = args.stage
  if stage == "train":
    df_train = pd.read_csv(train_dataset_path)
    # 5 fold to build the data
    kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
    train_encode = []
    for train_index, test_index in kf.split(df_train):
      df_train_encode, df_val = df_train.iloc[train_index], df_train.iloc[test_index]
      df_val = apply_feature_engineering(df_train, df_val)
      df_val = apply_target_encoding(df_train_encode, df_val)
      df_val.reset_index(inplace=True, drop=True)
      train_encode.append(df_val)
    
    df_train_encode = pd.concat(train_encode)
    df_train_encode.to_csv(os.path.join(save_data_path, "train_engineered.csv"), index=False)
  elif stage == "test":
    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)
    
    df_test = apply_feature_engineering(df_train, df_test)
    df_test = apply_target_encoding(df_train, df_test)

    df_test.to_csv(os.path.join(save_data_path, "test_engineered.csv"), index=False)

def apply_feature_engineering(df_train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
  return(
    df.copy()
    .pipe(add_category_features, category_ls = category_ls)
    .pipe(add_coe_date_features)
    .pipe(add_time_features)
    .pipe(add_make_model)
    .pipe(backfill_missing_cat_var)
    .pipe(backfill_arf)
    .pipe(backfill_coe, df_train=df_train)
    .pipe(backfill_dereg_value)
    .pipe(backfill_depreciation)
    .pipe(backfill_road_tax)
    .pipe(backfill_make_model_features, df_train=df_train)
    .pipe(backfill_mileage, df_train=df_train)
    .pipe(add_basic_text_features)
    .pipe(add_speicific_word_features)
    .pipe(add_one_hot_encode)
    .pipe(backfill_missing_num_var)
    )

def apply_target_encoding(df_train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
  return(
    df.copy()
    .pipe(cat_encoding, df_train=df_train)
  ) 

if __name__ == "__main__":
  main()