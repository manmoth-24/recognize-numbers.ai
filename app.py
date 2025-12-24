from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import os
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)

# --- 【重要】ここが抜けていました ---
# Renderなどのサーバーで動かすための設定
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
# --------------------------------

# --- AIモデルの読み込み ---
model_path = 'my_number_brain.keras'
if os.path.exists(model_path):
    # エラー回避のため、コンパイル不要で読み込む設定を追加
    model = keras.models.load_model(model_path, compile=False)
    print("AIモデル読み込み完了！")
else:
    print("エラー: 学習済みモデルが見つかりません。")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("リクエストを受け取りました...") # ログ用
    
    if model is None:
        print("エラー: モデルがロードされていません")
        return jsonify({'prediction': 'Error', 'confidence': 0})

    try:
        # 1. ブラウザから画像データを受け取る
        data = request.json['image']
        img_str = data.split(',')[1]
        
        # 2. 画像変換
        img_bytes = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img = img.resize((28, 28))
        
        # 3. 数値データに変換
        x = np.array(img) / 255.0
        x = x.reshape(1, 28, 28)
        
        print("予測を開始します...") # ログ用

        # 4. 予測実行
        prediction = model.predict(x)
        predicted_num = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        
        print(f"予測完了: {predicted_num}") # ログ用

        return jsonify({
            'prediction': int(predicted_num),
            'confidence': confidence
        })

    except Exception as e:
        print(f"エラー発生: {e}") # ログに詳細なエラーを出す
        return jsonify({'prediction': 'Error', 'confidence': 0})

if __name__ == '__main__':
    app.run(debug=True, port=5000)