import tkinter as tk
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# --- 1. 学習済みAIの読み込み ---
try:
    model = tf.keras.models.load_model('my_number_brain.keras')
    print("AIの読み込み完了！")
except:
    print("エラー: 'my_number_brain.keras' が見つかりません。先に学習を実行してください。")
    exit()

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手書き数字認識AI")

        # キャンバス設定（ここに描く）
        self.canvas_width = 300
        self.canvas_height = 300
        self.bg_color = "black" # AIの学習データに合わせて黒背景にする
        self.fg_color = "white" # 文字は白
        
        # Tkinterのキャンバス（表示用）
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg=self.bg_color)
        self.canvas.pack(pady=10)

        # PILの画像（AIに渡す用。キャンバスと同じ内容をメモリ内に描画する）
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)

        # マウスイベントの紐付け
        self.canvas.bind("<B1-Motion>", self.paint)

        # ボタン類
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        predict_btn = tk.Button(btn_frame, text="判定する！", font=("Arial", 14), command=self.predict_digit)
        predict_btn.pack(side=tk.LEFT, padx=10)
        
        clear_btn = tk.Button(btn_frame, text="消す", font=("Arial", 14), command=self.clear_canvas)
        clear_btn.pack(side=tk.LEFT, padx=10)

        # 結果表示ラベル
        self.result_label = tk.Label(root, text="ここに数字を書いてね", font=("Arial", 16))
        self.result_label.pack(pady=20)

    def paint(self, event):
        """マウスで線を描く処理"""
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        
        # 画面に描画
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.fg_color, outline=self.fg_color)
        
        # メモリ内の画像にも描画（AI用）
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear_canvas(self):
        """キャンバスをクリア"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="ここに数字を書いてね")

    def predict_digit(self):
        """AIによる判定処理"""
        # 1. 画像をAIが読めるサイズ(28x28)にリサイズする
        img_resized = self.image.resize((28, 28))
        
        # 2. numpy配列に変換 (0〜255)
        img_array = np.array(img_resized)
        
        # 3. 0.0〜1.0に正規化
        img_array = img_array / 255.0
        
        # 4. 形状を合わせる (1枚, 28, 28)
        img_array = img_array.reshape(1, 28, 28)
        
        # 5. AIで予測！
        predictions = model.predict(img_array)
        
        # 6. 一番確率が高い数字を取り出す
        predicted_number = np.argmax(predictions)
        confidence = np.max(predictions) * 100 # 自信度（%）

        self.result_label.config(text=f"これは... 「{predicted_number}」 です！\n(自信: {confidence:.1f}%)")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()