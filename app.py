from flask import Flask, render_template, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
import tensorflow as tf # 軽量版を使うためだけにインポート
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

# --- AIモデルの準備 (TFLite版) ---
tflite_path = 'model.tflite'
interpreter = None

if os.path.exists(tflite_path):
    try:
        # 軽量インタプリタの作成
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # 入力と出力の情報を取得しておく
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("軽量AIモデル(TFLite)読み込み完了！")
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
else:
    print("エラー: 'model.tflite' が見つかりません。")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({'prediction': 'Error', 'confidence': 0})

    try:
        # 1. 画像受け取り
        data = request.json['image']
        img_str = data.split(',')[1]
        img_bytes = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img = img.resize((28, 28))
        
        # 2. 数値変換 (ここが重要: float32型にする必要があります)
        x = np.array(img, dtype=np.float32) / 255.0
        x = x.reshape(1, 28, 28)

        # 3. 推論実行 (TFLite特有の書き方)
        # 入力データをセット
        interpreter.set_tensor(input_details[0]['index'], x)
        # 計算実行！
        interpreter.invoke()
        # 結果を取り出す
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_num = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100

        return jsonify({
            'prediction': int(predicted_num),
            'confidence': confidence
        })

    except Exception as e:
        print(f"エラー詳細: {e}")
        # エラーの中身をそのまま画面に返す
        return jsonify({'prediction': f'Err: {str(e)}', 'confidence': 0})

if __name__ == '__main__':
    app.run(debug=True, port=5000)