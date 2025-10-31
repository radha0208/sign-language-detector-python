import cv2
import pickle
import mediapipe as mp
import numpy as np
from collections import deque
import time
import tkinter as tk
from tkinter import simpledialog, scrolledtext, messagebox
from PIL import Image, ImageTk
import threading
from googletrans import Translator
from gtts import gTTS
from playsound import playsound
import tempfile

# ---------------- LOAD MODEL ----------------
with open('model.pkl', 'rb') as f:
    saved = pickle.load(f)
model = saved['model']
scaler = saved['scaler']
le = saved['label_encoder']

# ---------------- MEDIA PIPE ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- SETTINGS ----------------
ROLLING_FRAMES = 5
CONF_THRESHOLD = 0.7
ALPHABET_TIMEOUT = 1.0

pred_buffer = deque(maxlen=ROLLING_FRAMES)
current_word_buffer = []
last_alpha_time = None
current_sentence = ""
last_pred_label = None

translator = Translator()

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Sign Language Detection System")

# Layout frames
left_frame = tk.Frame(root)
left_frame.grid(row=0, column=0, padx=20, pady=20)

right_frame = tk.Frame(root)
right_frame.grid(row=0, column=1, padx=20, pady=20)

# Video feed (Left Center)
video_label = tk.Label(left_frame)
video_label.pack()

# Right-side UI elements neatly stacked
mode_var = tk.StringVar(value="Word Mode")
tk.Label(right_frame, text="Detection Mode:", font=("Arial", 11, "bold")).pack(anchor="w")
mode_menu = tk.OptionMenu(right_frame, mode_var, "Word Mode", "Alphabet Mode")
mode_menu.pack(fill="x", pady=5)

tk.Label(right_frame, text="Current Word:", font=("Arial", 11, "bold")).pack(anchor="w")
current_word_text = scrolledtext.ScrolledText(right_frame, width=35, height=4)
current_word_text.pack(pady=5)

tk.Label(right_frame, text="Current Sentence:", font=("Arial", 11, "bold")).pack(anchor="w")
current_sentence_text = scrolledtext.ScrolledText(right_frame, width=35, height=4)
current_sentence_text.pack(pady=5)


# ---------------- BUTTON FUNCTIONS ----------------
def clear_current_sentence():
    global current_sentence
    current_sentence = ""
    current_sentence_text.delete("1.0", tk.END)


def clear_last_word():
    global current_sentence
    words = current_sentence.strip().split(" ")
    if words:
        words.pop()
        current_sentence = " ".join(words) + " "
        current_sentence_text.delete("1.0", tk.END)
        current_sentence_text.insert(tk.END, current_sentence)


def play_audio_safely(path):
    """Play mp3 quietly in background without any popup"""
    def _play():
        try:
            playsound(path)
        except Exception as e:
            print(f"Audio error: {e}")
    threading.Thread(target=_play, daemon=True).start()


def speak_translated_text(text, lang_choice):
    """Speak the translated text in the chosen language"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts = gTTS(text=text, lang=lang_choice)
            tts.save(tmp.name)
            play_audio_safely(tmp.name)
    except Exception as e:
        messagebox.showerror("Error", f"Speaking failed: {e}")


def translate_detected():
    """Ask for translation language, show translated text, and speak option"""
    text = current_sentence_text.get("1.0", tk.END).strip()
    if not text:
        messagebox.showinfo("Info", "No text to translate.")
        return

    lang_choice = simpledialog.askstring(
        "Language",
        "Translate to which language? (ta, hi, ko, ja, de, fr)"
    )
    if not lang_choice:
        return

    try:
        trans = translator.translate(text, dest=lang_choice)
        translated_text = trans.text

        # Create popup for translation result
        popup = tk.Toplevel(root)
        popup.title(f"Translation ({lang_choice})")
        popup.geometry("400x220")
        popup.resizable(False, False)

        tk.Label(
            popup,
            text=f"Translated Text ({lang_choice}):",
            font=("Arial", 11, "bold")
        ).pack(pady=5)

        text_box = tk.Text(popup, wrap="word", height=4, width=40)
        text_box.insert("1.0", translated_text)
        text_box.config(state="disabled")
        text_box.pack(pady=5)

        # Speak button inside popup
        tk.Button(
            popup,
            text="🔊 Speak Translation",
            command=lambda: speak_translated_text(translated_text, lang_choice)
        ).pack(pady=10)

        tk.Button(popup, text="Close", command=popup.destroy).pack()

    except Exception as e:
        messagebox.showerror("Error", f"Translation failed: {e}")


# ---------------- BUTTONS ----------------
tk.Button(right_frame, text="Clear Current Sentence", command=clear_current_sentence).pack(pady=5, fill="x")
tk.Button(right_frame, text="Clear Last Word", command=clear_last_word).pack(pady=5, fill="x")
tk.Button(right_frame, text="Translate Detected", command=translate_detected).pack(pady=5, fill="x")


# ---------------- VIDEO LOOP ----------------
cap = cv2.VideoCapture(0)

def update_frame():
    global pred_buffer, current_word_buffer, last_alpha_time, current_sentence, last_pred_label
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            data_aux = []
            min_x, min_y = min(x_list), min(y_list)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)

            X_scaled = scaler.transform([data_aux])
            probs = model.predict_proba(X_scaled)[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            label = le.inverse_transform([pred_idx])[0]

            if confidence > CONF_THRESHOLD:
                pred_buffer.append(label)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if pred_buffer:
        pred_label = max(set(pred_buffer), key=pred_buffer.count)
        if pred_label != last_pred_label:
            last_pred_label = pred_label

            if mode_var.get() == "Word Mode" and pred_label.startswith("word_"):
                word_text = pred_label.replace("word_", "")
                current_sentence += word_text + " "
                current_word_text.delete("1.0", tk.END)
                current_word_text.insert(tk.END, word_text)

            elif mode_var.get() == "Alphabet Mode" and pred_label.startswith("alpha_"):
                current_word_buffer.append(pred_label.replace("alpha_", ""))
                current_word_text.delete("1.0", tk.END)
                current_word_text.insert(tk.END, "".join(current_word_buffer))
                last_alpha_time = time.time()

    if current_word_buffer and last_alpha_time:
        if time.time() - last_alpha_time > ALPHABET_TIMEOUT:
            word = "".join(current_word_buffer)
            current_sentence += word + " "
            current_word_buffer.clear()
            last_alpha_time = None
            current_word_text.delete("1.0", tk.END)

    current_sentence_text.delete("1.0", tk.END)
    current_sentence_text.insert(tk.END, current_sentence)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()