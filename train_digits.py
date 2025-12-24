import tensorflow as tf
import os

print("学習を開始します。しばらくお待ちください...")

# 1. 教科書データ（MNIST）をダウンロード
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. データを扱いやすく加工（0〜255の色情報を 0.0〜1.0 に変換）
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. AIの脳モデルを作成（ニューラルネットワーク）
model = tf.keras.models.Sequential([
    # 28x28マスの画像を、一列の数値(784個)に平らにする
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # 隠れ層：128個のニューロン（脳細胞）で特徴をつかむ
    tf.keras.layers.Dense(128, activation='relu'),
    
    # ドロップアウト：勉強しすぎ（過学習）を防ぐためにわざと少し忘れさせる
    tf.keras.layers.Dropout(0.2),
    
    # 出力層：0〜9の確率を出す（合計10個のニューロン）
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4. コンパイル（学習の方法を決める）
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. いざ学習実行！（5周くりかえす）
model.fit(x_train, y_train, epochs=5)

# 6. 成績発表
print("\n--- テストデータでの成績 ---")
model.evaluate(x_test,  y_test, verbose=2)

# 7. 脳をファイルに保存
model.save('my_number_brain.keras')
print("\n学習完了！ 'my_number_brain.keras' に保存しました。")