import os
from PIL import Image
def convert_and_resize_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".bmp"):
            # Открываем изображение
            img = Image.open(os.path.join(folder_path, filename))
            # Масштабируем до 28x28 пикселей
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            # Конвертируем в PNG и сохраняем
            new_filename = os.path.splitext(filename)[0] + ".png"
            new_folder_path = ('mnist_dataset/test_photo_12_12_png')
            img.save(os.path.join(new_folder_path, new_filename), "PNG")
convert_and_resize_images('mnist_dataset/test_photo_12_12_jpg')