from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64
import os
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)

# --- AIモデルの読み込み ---
model_path = 'my_number_brain.keras'
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)
    print("AIモデル読み込み完了！")
else:
    print("エラー: 学習済みモデルが見つかりません。")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'prediction': 'Error', 'confidence': 0})

    # 1. ブラウザから画像データを受け取る
    data = request.json['image']
    # データは "data:image/png;base64,....." という文字列できているので
    # コンマより後ろの実データ部分だけを取り出す
    img_str = data.split(',')[1]
    
    # 2. 文字列を画像として開く
    img_bytes = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_bytes)).convert('L') # グレースケール(白黒)に変換
    
    # 3. AIが読めるサイズ (28x28) にリサイズ
    img = img.resize((28, 28))
    
    # 4. 数値データに変換 (0.0〜1.0)
    x = np.array(img) / 255.0
    x = x.reshape(1, 28, 28) # 形を整える
    
    # 5. 予測実行
    prediction = model.predict(x)
    predicted_num = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100 # 信頼度(%)

    return jsonify({
        'prediction': int(predicted_num),
        'confidence': confidence
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)