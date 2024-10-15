import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# โหลดโมเดลที่บันทึกไว้
model = load_model('ai_recognition.h5')  # ใช้ชื่อไฟล์ตรง ๆ

# Dictionary สำหรับเก็บ embeddings
known_embeddings = {}

def load_known_embeddings(file_name):
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, index_col='name')
        for name, row in df.iterrows():
            known_embeddings[name] = row.values  # แปลงให้เป็น numpy array
    else:
        # สร้างไฟล์ CSV ถ้ายังไม่มี
        pd.DataFrame(columns=['name']).to_csv(file_name)

def compute_embedding(model, image_path):
    test_image = image.load_img(image_path, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    embedding = model.predict(test_image)
    return embedding.flatten()  # แปลงเป็นเวกเตอร์ 1 มิติ

def add_folder():
    folder_selected = filedialog.askdirectory()  # เปิดหน้าต่างเลือกโฟลเดอร์
    if folder_selected:
        name = simpledialog.askstring("Input", "Enter the name for this face:")  # ขอชื่อจากผู้ใช้
        if not name:
            messagebox.showwarning("Warning", "Name cannot be empty.")
            return

        embeddings_list = []
        for image_file in os.listdir(folder_selected):
            image_path = os.path.join(folder_selected, image_file)

            try:
                # โหลดเวกเตอร์ embedding ของรูปภาพ
                embedding = compute_embedding(model, image_path)
                embeddings_list.append(embedding)
            except (IOError, OSError, ValueError) as e:
                print(f"Skipping file {image_file}: {e}")  # แสดงข้อผิดพลาดในคอนโซล
                continue  # ข้ามไปยังไฟล์ถัดไป

        if embeddings_list:  # ตรวจสอบว่ามี embeddings สำหรับชื่อที่ระบุ
            # คำนวณค่าเฉลี่ยของ embeddings สำหรับใบหน้านี้
            known_embeddings[name] = np.mean(embeddings_list, axis=0)

            # เซฟข้อมูลลงไฟล์ CSV
            save_embeddings_to_csv('known_embeddings.csv')

            messagebox.showinfo("Success", f"Add {name} success")
        else:
            messagebox.showwarning("Warning", f"No valid images found for {name}.")
        
        main_menu()

def save_embeddings_to_csv(file_name):
    # สร้าง DataFrame จาก known_embeddings
    df = pd.DataFrame.from_dict(known_embeddings, orient='index')
    df.columns = [f'feature_{i}' for i in range(df.shape[1])]  # ตั้งชื่อคอลัมน์
    df.index.name = 'name'  # ตั้งชื่อ index

    # บันทึก DataFrame ลง CSV
    df.to_csv(file_name)

def predict_face():
    file_selected = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])  # เลือกไฟล์รูป
    if file_selected:
        try:
            # โหลดและเตรียมรูปภาพ
            uploaded_embedding = compute_embedding(model, file_selected)
            result = recognize_face(uploaded_embedding)  # ใช้ฟังก์ชัน recognize_face ที่คุณสร้างไว้

            # แสดงผลการทำนาย
            show_image_with_message(Image.open(file_selected), result)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def recognize_face(uploaded_embedding):
    for name, known_embedding_vector in known_embeddings.items():
        if is_recognized(known_embedding_vector, uploaded_embedding):
            return f"You are {name}"
    return "Unknown person"

def is_recognized(known_embedding, unknown_embedding, threshold=0.2):
    distance = find_euclidean_distances(known_embedding, unknown_embedding)
    similarity = find_cosine_similarity(known_embedding, unknown_embedding)
    if distance < threshold and similarity > 0.98:
        return True  # ใกล้เคียง
    else:
        return False  # ไม่รู้จัก

def find_euclidean_distances(x, y):
    x = np.array(x)
    y = np.array(y)
    diff = x - y
    distance = np.sqrt(np.sum(diff**2))
    return distance

def find_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return cosine_similarity

def show_image_with_message(img, message):
    for widget in root.winfo_children():
        widget.destroy()

    img = img.resize((400, 400))  # ขยายขนาดรูปภาพให้ใหญ่ขึ้น
    img_tk = ImageTk.PhotoImage(img)
    img_label = tk.Label(root, image=img_tk, bg='#f0f0f0')
    img_label.image = img_tk
    img_label.pack(pady=10, expand=True)

    message_label = tk.Label(root, text=message, font=("Helvetica", 14), bg='#f0f0f0', fg='#333')
    message_label.pack(pady=5, expand=True)

    button_frame = tk.Frame(root, bg='#f0f0f0')
    button_frame.pack(pady=10, expand=True)

    back_button = tk.Button(button_frame, text="Back to Main Menu", command=main_menu, font=("Helvetica", 12), bg='#4CAF50', fg='white', bd=0)
    back_button.pack(pady=5, padx=10)

def main_menu():
    for widget in root.winfo_children():
        widget.destroy()

    root.configure(bg='#f0f0f0')

    add_button = tk.Button(root, text="Add New Face", command=add_folder, font=("Helvetica", 12), bg='#2196F3', fg='white', bd=0)
    add_button.pack(pady=5, expand=True)

    predict_button = tk.Button(root, text="Predict Face", command=predict_face, font=("Helvetica", 12), bg='#2196F3', fg='white', bd=0)
    predict_button.pack(pady=5, expand=True)

    exit_button = tk.Button(root, text="Exit", command=root.quit, font=("Helvetica", 12), bg='#f44336', fg='white', bd=0)
    exit_button.pack(pady=5, expand=True)

# สร้างหน้าต่างหลัก
root = tk.Tk()
root.title("AI Model Test")
root.attributes('-fullscreen', True)  # ทำให้โปรแกรมเป็นโหมดเต็มหน้าจอ
root.configure(bg='#f0f0f0')

# โหลด embeddings ที่มีอยู่ก่อนหน้า
load_known_embeddings('known_embeddings.csv')

# เรียกเมนูหลักตอนเริ่มต้น
main_menu()

# เริ่มการทำงานของ Tkinter
root.mainloop()
