from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def preprocess_image(opencv_image) :
    # 光照均衡化
    equalized_image = cv2.equalizeHist(opencv_image)
    # Gamma 校正
    gamma = 1.5
    gamma_corrected_image = np.uint8(cv2.pow(equalized_image / 255.0, gamma) * 255)
    # 高斯模糊
    blurred_image = cv2.GaussianBlur(gamma_corrected_image, (5, 5), 0)
    # 適應性閾值處理
    adaptive_thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -15)
    # 色彩反轉
    inverted_image = cv2.bitwise_not(adaptive_thresh)
    # 去噪
    denoised_image = cv2.fastNlMeansDenoising(inverted_image, None, 30, 7, 21)

    return denoised_image

def preprocess_image2(opencv_image) :
    # 色彩反轉
    image = cv2.bitwise_not(opencv_image)
    try:
        # 適應性閾值處理
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 15)
    except:
        pass
    # 去噪
    try:
        denoised_image = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
    except:
        denoised_image = image

    return denoised_image

def preprocess_image3(opencv_image) :
    # Gamma 校正
    gamma = 1.5
    denoised_image = np.uint8(cv2.pow(opencv_image / 255.0, gamma) * 255)

    return denoised_image

def detect(image):
    # 設定 Tesseract 的 psm 和 oem 參數
    custom_config = r'--psm 6 --oem 2'

    # 轉換為灰階圖像
    gray_image = image.convert('L')

    # 計算圖片的光線值
    calc_brig = np.mean(gray_image)

    # 設定 Tesseract 可執行文件的路徑 (Windows)
    # pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

    # 影像預處理
    # 轉換為 OpenCV 影像
    opencv_image = np.array(gray_image)

    # 根據光線值進行條件判斷並採取相應的操作
    if calc_brig > 150:
        # print("圖片光線值大於150，進行預處理。")
        denoised_image = preprocess_image(opencv_image)
    elif calc_brig < 50 :
        # 圖片光線值小於50
        denoised_image = preprocess_image3(opencv_image)
    else:
        # print("圖片光線值小於150和大於50，進行預處理2。")
        denoised_image = preprocess_image2(opencv_image)

    # 轉換回 PIL 圖像
    preprocessed_image = Image.fromarray(denoised_image)

    text = pytesseract.image_to_string(preprocessed_image, lang='eng', config=custom_config) # Eng為英文
    # 允許的字元
    allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    # 把所有英文字改成大寫，並刪除不在allowed_chars裡面的字
    text_filtered = ''.join(filter(allowed_chars.__contains__, text)).upper()
    # 返回辨識出的文字
    return text_filtered

def get_text_from_image(image):
    if hasattr(image, 'shape'):
        image = Image.fromarray(image)
    text = detect(image)
    text = text.replace(" ", "")
    return text

if __name__ == '__main__':
    image = Image.open(r'../yolo/image_0001.jpg')
    crop_image = image.crop((0, 0, 1000, 1000))
    text = get_text_from_image(crop_image)
    print(text)