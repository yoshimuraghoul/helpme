import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import pywt


class ImageProcessor:
    def __init__(self, master):
        self.master = master
        master.title("Wavelet Thresholding")

        # Строки пути к входному и выходному изображениям
        self.input_image_path = tk.StringVar()
        self.output_image_path = ""

        # Создание виджетов для загрузки входного изображения
        self.select_input_image_label = ttk.Label(master, text="Выберите входное изображение:")
        self.select_input_image_label.grid(row=0, column=0, pady=10)

        self.browse_input_image_button = ttk.Button(master, text="Обзор...", command=self.load_input_image)
        self.browse_input_image_button.grid(row=0, column=1, pady=10)

        self.input_image_path_label = ttk.Label(master, textvariable=self.input_image_path)
        self.input_image_path_label.grid(row=1, column=0, columnspan=2)

        # Создание виджетов для выбора параметров обработки изображения
        self.scale_label = ttk.Label(master, text="Масштабирование:")
        self.scale_label.grid(row=2, column=0)

        self.scale = tk.Scale(master, from_=0.1, to=1.0, orient=tk.HORIZONTAL, resolution=0.1)
        self.scale.grid(row=2, column=1)

        self.brightness_label = ttk.Label(master, text="Коррекция яркости:")
        self.brightness_label.grid(row=3, column=0)

        self.brightness = tk.Scale(master, from_=-50, to=50, orient=tk.HORIZONTAL)
        self.brightness.grid(row=3, column=1)

        self.contrast_label = ttk.Label(master, text="Коррекция контраста:")
        self.contrast_label.grid(row=4, column=0)

        self.contrast = tk.Scale(master, from_=0.1, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
        self.contrast.grid(row=4, column=1)

        self.threshold_label = ttk.Label(master, text="Порог для Wavelet Thresholding:")
        self.threshold_label.grid(row=5, column=0)

        self.threshold = tk.Scale(master, from_=0.1, to=1.0, orient=tk.HORIZONTAL, resolution=0.05)
        self.threshold.grid(row=5, column=1)

        # Создание кнопки для обработки изображения
        self.process_image_button = ttk.Button(master, text="Обработать изображение", command=self.process_image)
        self.process_image_button.grid(row=6, column=0, columnspan=2, pady=(20, 10))

        # Создание виджетов для просмотра выходного изображения
        self.show_output_image_button = ttk.Button(master, text="Показать результат", command=self.show_output_image, state=tk.DISABLED)
        self.show_output_image_button.grid(row=7, column=0)

        self.output_image_path_label = ttk.Label(master, text="")
        self.output_image_path_label.grid(row=7, column=1)

        self.output_label = ttk.Label(master)
        self.output_label.grid(row=8, column=0, columnspan=2)

        # Создание виджетов для сохранения выходного изображения
        self.save_output_image_button = ttk.Button(master, text="Сохранить результат", command=self.save_output_image, state=tk.DISABLED)
        self.save_output_image_button.grid(row=9, column=0, columnspan=2, pady=(10, 20))

    def load_input_image(self):
        # Открытие диалогового окна для выбора файла
        file_types = (("JPEG files", "*.jpg"), ("All files", "*.*"))
        file_path = filedialog.askopenfilename(title="Выберите входное изображение", filetypes=file_types)

        if not file_path:
            return

        # Загрузка изображения и его отображение
        self.input_image = Image.open(file_path)
        self.input_image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(self.input_image)
        self.input_label.configure(image=photo)
        self.input_label.image = photo

        # Обновление пути к входному изображению
        self.input_image_path.set(file_path)

        # Активация кнопки для обработки изображения
        self.process_image_button.configure(state=tk.NORMAL)

    def process_image(self):
        if self.input_image:
            # Масштабирование изображения
            w, h = self.input_image.size
            new_w, new_h = int(w*self.scale.get()), int(h*self.scale.get())
            resized_img = self.input_image.resize((new_w, new_h))

            # Корректирование яркости и контраста изображения
            brightness = self.brightness.get()
            contrast = self.contrast.get()
            img_array = cv2.cvtColor(cv2.imread(self.input_image_path.get()), cv2.COLOR_BGR2RGB)
            adjusted = cv2.convertScaleAbs(img_array, beta=brightness)
            adjusted = cv2.convertScaleAbs(adjusted, alpha=contrast, beta=brightness)

            # Применение метода Wavelet Thresholding
            wavelet = 'db4'
            mode = 'soft'
            threshold = self.threshold.get()

            coeffs2 = pywt.dwt2(adjusted, wavelet)
            max_coeff = max(coeffs2[0].max(), coeffs2[1].max(), coeffs2[2].max())
            if max_coeff <= 1e-10:
                # Если максимальный коэффициент равен нулю, то ничего не делаем
                denoised_img = adjusted
            else:
                # Иначе, применяем thresholding
                coeffs2 = pywt.threshold(coeffs2, threshold*max_coeff, mode)
                denoised_img = pywt.idwt2(coeffs2, wavelet)

            # Удаление шумов при помощи медианного фильтра
            denoised_img = cv2.medianBlur(denoised_img.astype(np.uint8), 5)

            # Сохранение выходного изображения
            output_path = os.path.splitext(self.input_image_path.get())[0] + "_output.jpg"
            self.output_image_path = output_path
            Image.fromarray(denoised_img).save(output_path)

            # Вывод пути к выходному изображению
            self.output_image_path_label.configure(text=output_path)

            # Вывод выходного изображения
            self.output_image = Image.fromarray(denoised_img)
            self.output_image.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(self.output_image)
            self.output_label.configure(image=photo)
            self.output_label.image = photo
            self.master.update()

            # Активация кнопок для просмотра и сохранения выходного изображения
            self.show_output_image_button.configure(state=tk.NORMAL)
            self.save_output_image_button.configure(state=tk.NORMAL)

    def show_output_image(self):
        # Отображение выходного изображения во всплывающем окне
        if self.output_image:
            img_viewer = tk.Toplevel(self.master)
            img_viewer.title("Результат")
            photo = ImageTk.PhotoImage(self.output_image)
            img_viewer_label = ttk.Label(img_viewer, image=photo)
            img_viewer_label.image = photo
            img_viewer_label.pack()

    def save_output_image(self):
        # Выбор места сохранения выходного изображения и его сохранение
        file_types = (("JPEG files", "*.jpg"), ("All files", "*.*"))
        file_path = filedialog.asksaveasfilename(title="Сохранить выходное изображение", filetypes=file_types,
                                                 defaultextension=".jpg")

        if not file_path:
            return

        self.output_image.save(file_path)

    def start(self):
        # Создание виджета для входного изображения
        self.input_label = ttk.Label(self.master)
        self.input_label.grid(row=0, column=2, rowspan=3)

        # Создание виджетов для выбора параметров обработки изображения
        ttk.Label(self.master, text="Параметры обработки изображения:").grid(row=2, column=2, pady=(0, 10))

        ttk.Label(self.master, text="Масштабирование:").grid(row=3, column=2, sticky="e")
        ttk.Label(self.master, text="Коррекция яркости:").grid(row=4, column=2, sticky="e")
        ttk.Label(self.master, text="Коррекция контраста:").grid(row=5, column=2, sticky="e")
        ttk.Label(self.master, text="Порог для Wavelet Thresholding:").grid(row=6, column=2, sticky="e")


def main():
    root = tk.Tk()
    app = ImageProcessor(root)
    app.start()
    root.mainloop()


if __name__ == '__main__':
    main()
