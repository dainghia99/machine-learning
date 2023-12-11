import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score

model = load_model('emotion_detection.h5')

test_dir = 'anhTest'
output_dir = 'anhDoanDung'
ouput_error = 'anhDoanSai'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class_labels = ['tucgian', 'ghetom', 'loso', 'vuive', 'buon', 'ngacnhien', 'binhthuong']

so_anh_dung = 0
tong_so_anh = 0

for emotion_folder in os.listdir(test_dir):
    emotion_path = os.path.join(test_dir, emotion_folder)
    if os.path.isdir(emotion_path):
        for filename in os.listdir(emotion_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(emotion_path, filename)
                img = cv2.imread(img_path)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, (48, 48))
                img_normalized = img_resized / 255.0
                img_input = np.expand_dims(img_normalized, axis=-1)

                predictions = model.predict(np.array([img_input]))
                predicted_label = class_labels[predictions.argmax()]



                label_file = filename.split('-')
                name_label = label_file[0]
                
                if predicted_label == name_label:
                    so_anh_dung += 1
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    font_thickness = 1
                    text = f'{predicted_label}'
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    text_position = (10, 35)

                    cv2.putText(img, text, text_position, font, font_scale, (245, 57, 57), font_thickness, cv2.LINE_AA)

                    output_path = os.path.join(output_dir, f'{emotion_folder}_{filename}')
                    cv2.imwrite(output_path, img)
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    font_thickness = 1
                    text = f'{predicted_label}'
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    text_position = (10, 35)

                    cv2.putText(img, text, text_position, font, font_scale, (245, 57, 57), font_thickness, cv2.LINE_AA)

                    output_path = os.path.join(ouput_error, f'{emotion_folder}_{filename}')
                    cv2.imwrite(output_path, img)


                tong_so_anh += 1

                

print(f'Số ảnh đúng: {so_anh_dung}')
print(f'Tổng số ảnh training: {tong_so_anh}')

ketqua = (so_anh_dung / tong_so_anh) * 100

print("Kết quả là: ", round(ketqua, 2), "%")
