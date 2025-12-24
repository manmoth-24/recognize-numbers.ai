import tensorflow as tf

# 1. 今ある重たいAIを読み込む
print("ロード中...")
model = tf.keras.models.load_model('my_number_brain.keras')

# 2. 軽量版(TFLite)に変換するコンバーターを準備
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. 変換実行！
print("変換中...")
tflite_model = converter.convert()

# 4. ファイルに保存
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("完了！ 'model.tflite' ができました。")