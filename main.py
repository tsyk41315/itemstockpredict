from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date
import pandas as pd
import numpy as np
import xgboost as xgb
import os

print("カレントディレクトリ:", os.getcwd())

# インスタンス化
app = FastAPI()


# モデルファイルの正確な絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(current_dir, 'models', 'stockpredict.json')



if os.path.exists(model_file_path):
    print("モデルファイルが存在します。")
else:
    print("モデルファイルが見つかりません。パス:", model_file_path)


# XGBoostモデルをロード
bst = xgb.Booster()
if os.path.exists(model_file_path):
    bst.load_model(model_file_path)
else:
    raise FileNotFoundError(f"{model_file_path} が見つかりません。")

class UserInput(BaseModel):
    item_id: int
    start_date: date
    end_date: date


bst.load_model(model_file_path)

@app.get('/predict/{item_id}/{start_date}/{end_date}')
async def predict_sales(item_id: int, start_date: str, end_date: str):
    # データの読み込みと前処理
    train = pd.read_csv('train.csv', parse_dates=['date'])
    
    # ユーザーが選択した期間でフィルタリング
    filtered_df = train[(train['item'] == item_id) &
                        (train['date'] >= pd.to_datetime(start_date)) &
                        (train['date'] <= pd.to_datetime(end_date))]

    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="Data not found for the given period and item ID.")
    
    # 未来の売上を予測するための特徴量を準備
    X_future = []
    sales_values = filtered_df['sales'].values
    for i in range(365, len(sales_values)):
        X_future.append(sales_values[i-365:i])
    X_future = np.array(X_future)

    # XGBoostのDMatrixを作成
    # 注: この例ではX_futureの形状や前処理がモデルの要求と合致することを前提としています
    # 実際には、モデルに適した特徴量の形状に変換する必要があります
    dtest = xgb.DMatrix(X_future)

    # 予測を実行
    predictions = bst.predict(dtest)

    
    return {
        "item_id": item_id,
        "predictions": predictions,
        "start_date": start_date,
        "end_date": end_date
    }
