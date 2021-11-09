import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.utils.config import load_config
from src.models.regressor import SklearnRegressor, CategoricalBoostRegressor

PROJECT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
train_engineered_path = os.path.join(PROJECT_DIR, "data/processed/train_engineered.csv")
test_engineered_path = os.path.join(PROJECT_DIR, "data/processed/test_engineered.csv")

def main():
  config = load_config()
  target = config["target"]

  df_train = pd.read_csv(train_engineered_path)
  df_test = pd.read_csv(test_engineered_path)

  print ("training random forest model")
  rf_param_dic = config["rf_parameters"]
  rf_features = config["rf_selected_features"]

  rf_estimator = RandomForestRegressor(**rf_param_dic)

  rf_model = SklearnRegressor(rf_estimator, 
                              rf_features, 
                              target)
  rf_model.train(df_train)

  y_pred = rf_model.predict(df_test[rf_features])
  df_predict_rf = pd.DataFrame({"Predicted": y_pred})
  df_predict_rf.to_csv(os.join(PROJECT_DIR, "data/processed/predict_rf.csv", index_label = "Id"))

  print ("training xgboost model")
  xgboost_param_dic = config["xgboost_parameters"]
  xgboost_estimator = XGBRegressor(**xgboost_param_dic)
  xgboost_features = config["num_features"] + config["cat_encoding_features"] + config["cat_feature_encoding"]

  xgboost_model = SklearnRegressor(xgboost_estimator, xgboost_features, target)
  xgboost_model.train(df_train)

  y_pred = xgboost_model.predict(df_test[xgboost_features])
  df_predict_xgboost = pd.DataFrame({"Predicted": y_pred})
  df_predict_xgboost.to_csv(os.join(PROJECT_DIR, "data/processed/predict_xgboost.csv", index_label = "Id"))

  print ("training catboost model")
  catboost_param_dic = config["catboost_parameters"]
  catboost_features = config["num_features"] + config["cat_features"]
  catboost_features_with_target = catboost_features + [target]
  df_selected_catboost = df_train[catboost_features_with_target]
  cat_features_loc = [df_selected_catboost.columns.get_loc(col) for col in config["cat_features"]]

  catboost_estimator = CatBoostRegressor(**catboost_param_dic)
  catboost_model = CategoricalBoostRegressor(catboost_estimator, catboost_features, target, cat_features_loc)
  catboost_model.train(df_train[catboost_features_with_target])

  y_pred = catboost_model.predict(df_test[catboost_features_with_target])
  df_predict_catboost = pd.DataFrame({"Predicted": y_pred})
  df_predict_catboost.to_csv(os.join(PROJECT_DIR, "data/processed/predict_catboost.csv", index_label = "Id"))

if __name__ == "__main__":
    main()
