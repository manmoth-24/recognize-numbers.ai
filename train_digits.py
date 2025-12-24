import tensorflow as tf
import os

# データセット読み込み
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# モデル定義
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# コンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 学習実行
model.fit(x_train, y_train, epochs=5)

# 保存（ここが重要！）
model.save('my_number_brain.keras')
print("完了！ 'my_number_brain.keras' を保存しました。")