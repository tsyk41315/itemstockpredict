import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import streamlit as st


# 相対パスの指定（例: スクリプトと同じディレクトリにある場合）
file_path = 'train.csv'

 
# データ読み込みと前処理を行う関数
@st.cache_data

def load_and_preprocess_data():
    # train.csvからデータを読み込む
    train = pd.read_csv('train.csv', parse_dates=['date'])
    
    # 2013年から2016年までのデータをフィルタリング
    df = train[(train['date'].dt.year >= 2013) & (train['date'].dt.year <= 2016)]
    
    # アイテムIDが1, 2, 3のデータを選択
    df = df[df['item'].isin([1, 2, 3])]

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    return df

df = load_and_preprocess_data()

def prepare_prediction_data():
    # train.csvからデータを読み込む
    train = pd.read_csv('train.csv', parse_dates=['date'])
    
    # 2017年のデータをフィルタリング
    df_2017 = train[train['date'].dt.year == 2017]
    
    # アイテムIDが1, 2, 3のデータを選択
    df_2017 = df_2017[df_2017['item'].isin([1, 2, 3])]
    
    # ここで、予測に必要な特徴量を選択または生成します
    # 例: df_2017 = df_2017[['feature1', 'feature2']]
    
    return df_2017

df_2017 = prepare_prediction_data

# 特徴量とラベルを生成する関数
def generate_features_and_labels(df, user_input):
    filtered_df = df[df['item'] == user_input]
    X = []
    y = []
    for i in range(365, len(filtered_df)):
        X.append(filtered_df['sales'].values[i-365:i])
        y.append(filtered_df['sales'].values[i])
    X = np.array(X)
    y = np.array(y)
    return X, y


# ユーザー入力
user_input = st.number_input("Enter the item ID:", min_value=1, max_value=3, value=1)

# 特徴量とラベルの生成
X, y = generate_features_and_labels(df, user_input)

# トレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DMatrix の作成
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

#トレーニングのパラメータ
param = {
    'max_depth': 3,
    'eta': 0.3,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
    }
num_round = 100
bst = xgb.train(param, dtrain, num_round, evals=[(dtrain, 'train'), (dtest, 'eval')], early_stopping_rounds=10)
    

# 予測
y_pred = bst.predict(dtest)

# ここでモデルのダンプと特徴量の重要度を取得
dump_info = bst.get_dump()
print(dump_info[0])  # 最初の決定木を表示
feature_importance = bst.get_score(importance_type='weight')
print(feature_importance)

print("カレントディレクトリ:", os.getcwd())

model_file_dir = './models'
model_file_path = os.path.join(model_file_dir, 'stockpredict.json')

# ディレクトリの作成確認
if not os.path.exists(model_file_dir):
    print(f"{model_file_dir} ディレクトリが存在しません。作成します。")
    os.makedirs(model_file_dir)
else:
    print(f"{model_file_dir} ディレクトリは既に存在します。")

# モデルの保存
print(f"モデルを {model_file_path} に保存します。")
bst.save_model(model_file_path)
print("モデルの保存が完了しました。")

