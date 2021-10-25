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
  backfill_mileage
)

from src.features.encoding import (
  cat_encoding
)


train_dataset_path = "/Users/user/Desktop/cs5228_project_MARS/data/raw/train.csv"
category_ls = generate_categories(pd.read_csv(train_dataset_path))


def main():
  argparser = ArgumentParser()
  argparser.add_argument('--stage', type=str, required=True)
  args = argparser.parse_args()

  stage = args.stage
  if stage == "train":
    df_train = pd.read_csv("/Users/user/Desktop/cs5228_project_MARS/data/raw/train.csv")
    # 5 fold to build the data
    kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
    train_encode = []
    for train_index, test_index in kf.split(df_train):
      df_train_encode, df_val = df_train.iloc[train_index], df_train.iloc[test_index]
      df_val = apply_feature_engineering(df_val)
      df_val = apply_imputation(df_train_encode, df_val)
      df_val = apply_target_encoding(df_train_encode, df_val)
      df_val.reset_index(inplace=True, drop=True)
      train_encode.append(df_val)
    
    df_train_encode = pd.concat(train_encode)
    df_train_encode.to_csv("/Users/user/Desktop/cs5228_project_MARS/data/processed/train_engineered.csv")
  elif stage == "test":
    df_train = pd.read_csv("/Users/user/Desktop/cs5228_project_MARS/data/raw/train.csv")
    df_test = pd.read_csv("/Users/user/Desktop/cs5228_project_MARS/data/raw/test.csv")

    df_test = apply_feature_engineering(df_test)
    df_test = apply_imputation(df_train, df_test)
    df_test = apply_target_encoding(df_train, df_test)

    df_test.to_csv("/Users/user/Desktop/cs5228_project_MARS/data/processed/test_engineered.csv")

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
  return(
    df.copy()
    .pipe(add_category_features, category_ls = category_ls)
    .pipe(add_basic_text_features)
    .pipe(add_speicific_word_features)
    .pipe(add_one_hot_encode)
    .pipe(add_make_model)
    .pipe(add_coe_date_features)
    .pipe(add_time_features))

def apply_imputation(df_train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
  return(
    df.copy()
    .pipe(backfill_arf)
    .pipe(backfill_coe, df_train=df_train)
    .pipe(backfill_dereg_value)
    .pipe(backfill_depreciation)
    .pipe(backfill_road_tax)
    .pipe(backfill_make_model_features, df_train=df_train)
    .pipe(backfill_mileage, df_train=df_train)
  )

def apply_target_encoding(df_train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
  return(
    df.copy()
    .pipe(cat_encoding, df_train=df_train)
  ) 

if __name__ == "__main__":
  main()