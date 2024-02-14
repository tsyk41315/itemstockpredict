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
st.image(image, caption='倉庫の画像', use_column_width=True)


# アイテム選択のためのexpander
expander1 = st.expander('どのアイテムの予測をしますか？')
with expander1:
    # ユーザーにアイテムIDを入力させる
    item_id = st.text_input("Enter the item ID:")

expander2 = st.expander('予測期間の開始日は？')
with expander2:
    # ユーザーに予測期間の開始日を入力させる
    start_date = st.date_input("Select the start date:")

expander3 = st.expander('予測期間の終了日は？')
with expander3:
    # ユーザーに予測期間の開始日を入力させる
    end_date = st.date_input("Select the end date:")


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
            st.write(f"Predicted values: {predictions['predicted_sales']}")
        else:
            st.write("Error in prediction")
    else:
        st.error("Please fill in all fields.")
    
# モデルの読み込み
bst_loaded = xgb.Booster()
bst_loaded.load_model('stockpredict.json')


import time

latest_iteration = st.empty()
bar = st.progress(0)

# プログレスバーを0.1秒毎に進める
for i in range(100):
    latest_iteration.text(f'Iteration{i + 1}')
    bar.progress(i + 1)
    time.sleep(0.1)

'Done'





