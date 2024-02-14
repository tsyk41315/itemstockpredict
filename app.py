# 必要なライブラリをインポート
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import xgboost as xgb


# タイトルとテキストを記入
st.title('在庫予測システム')
st.write('指定期間を選択すると期間内に必要な数量を算出します')

# Pillow()
from PIL import Image


# 画像ファイルを開く
image_path = '/Users/yasuhirokishi/Desktop/souko.png'  # 画像ファイルのパスを指定
image = Image.open(image_path)

# 画像を表示
st.image(image, use_column_width=True)


# アイテム選択のためのexpander
expander = st.expander('Iterationが100になったら押してください')
with expander:
    item_id = st.text_input("どのアイテムの予測をしますか？")  # アイテムIDの入力
    start_date = st.date_input("予測期間の開始日を選択してください:")  # 開始日の選択
    end_date = st.date_input("予測期間の終了日を選択してください:")  # 終了日の選択

# Streamlitウィジェットを使用してユーザー入力を取得
#item_id = st.number_input("Enter the item ID:", min_value=1, max_value=3, value=1)
#start_date = st.date_input("Select the start date:")
#end_date = st.date_input("Select the end date:")


# 予測ボタン
if st.button('Predict') and start_date and end_date:
    # 入力がすべて完了しているか確認
    if item_id and start_date and end_date:
    # FastAPIエンドポイントにリクエストを送信
        url = f"http://127.0.0.1:8000/predict/{item_id}/{start_date}/{end_date}"
        response = requests.get(url)
    
        if response.status_code == 200:
            # 予測結果の表示
            predictions = response.json()
            # 予測値のリストから合計値を計算
            total_predicted_sales = sum(predictions['predictions'])
            
            # 合計値を整数に丸める（必要に応じて）
            total_predicted_sales_rounded = round(total_predicted_sales)
            
            # 合計予測値の表示
            st.write(f"Predicted total sales from {start_date} to {end_date}: {total_predicted_sales_rounded}")
        else:
            st.write("Error in prediction")
    else:
        st.error("Please fill in all fields.")
    
# モデルの読み込み
bst_loaded = xgb.Booster()
bst_loaded.load_model('models/stockpredict.json')



import time

latest_iteration = st.empty()
bar = st.progress(0)

# プログレスバーを0.1秒毎に進める
for i in range(100):
    latest_iteration.text(f'Iteration{i + 1}')
    bar.progress(i + 1)
    time.sleep(0.1)

'Done'





