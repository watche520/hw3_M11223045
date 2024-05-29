# 機器學習 專案作業三 (第十組)

依賴套件安裝:

1. 根據作業系統安裝Tesseract-OCR

Linux安裝指令

```shell
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-eng
```

2. 根據[GitHub: AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)說明進行Darknet編譯  
>Linux能夠嘗試使用`Issue-2\yolo\libdarknet.so`直接執行`Issue-2\yolo.ipynb`

3. Python套件

```shell
pip install -r requirements.txt
```

## 權重檔

| 檔名(下載連結) | 描述 | 放置路徑 |
| -- | -- | -- |
| [best_model.pth](https://liveyuntechedu-my.sharepoint.com/:u:/g/personal/m11223010_live_yuntech_edu_tw/EZ850HAw979JiWhoGTXWNwoByuMz0zYChGoaL7Wbrbr3Fg?e=fyvQIQ) | Faster RCNN模型權重檔 | `Issue-2\fastercnn\weights\best_model.pth` |
| [yolov4-custom_last.weights](https://liveyuntechedu-my.sharepoint.com/:u:/g/personal/m11223010_live_yuntech_edu_tw/Ec5r-CfSO85Lq2D6QdL1irIBZ4Aq8xv-weRU21bEv2hf0g?e=YRTXYC) | YOLOv4模型權重檔 | `Issue-2\yolo\weights\yolov4-custom_last.weights` |

## 專案架構

```
|-- Issue-1: 議題一
|   |-- Faster_RCNN.ipynb: 訓練Faster RCNN模型
|   |-- OCR_&_Preprocess.ipynb: Tesseract-OCR實作與編號區域影像處理
|   |-- SSD.ipynb: SSD模型訓練及辨識
|   |-- YOLOv4_crop_img.ipynb: 使用YOLOv4和訓練好的模型切割影像
|   |-- YOLOv4_train.ipynb: 編譯Darknet環境、訓練YOLOv4模型
|
|-- Issue-2: 議題二
|   |-- 影片資料集: 預設讀取影片及保存輸出影片目錄
|   |   |...
|   |-- fastercnn: Faster RCNN套件
|   |   |...
|   |-- yolo: yolo套件與相關函數
|   |   |...
|   |-- faster_rcnn.ipynb: 使用Faster RCNN和Tesseract-OCR進行影片的編號辨識
|   |-- ocr.py: 供faster_rcnn.ipynb與yolo.ipynb共同引用的OCR套件，包含Tesseract-OCR實作與編號區域影像處理
|   |-- yolo.ipynb: 使用YOLOv4和Tesseract-OCR進行影片的編號辨識
|
|-- README.md
|-- requirements.txt: Python依賴套件列表
```


## 組員

* 資訊管理系-M11223045-王仁宏
* 資訊管理系-M11223010-葉哲丞
* 資訊管理系-M11223044-徐嘉佑
* 資訊管理系-M11223050-廖子皓

# 基於深度學習技術辨識貨櫃號碼

本研究旨在探討如何使用深度學習實作貨櫃號碼的自動化辨識。在議題一中，我們對單張影像進行處理，採用了YOLO、Faster RCNN 和 SSD 這三種物件偵測模型進行貨櫃號碼區域的定位，再結合Tesseract OCR 進行文字的辨識。實驗結果表示 YOLO 和 Faster RCNN 在物件偵測的情況下表現良好，但SSD無法偵測到預期目標。在議題二中，我們將這些技術應用於影片的逐幀辨識。然而，本實驗的文字辨識部分準確率仍有進步空間，導致最後產出的辨識號碼無法完全正確的輸出。我們發現場景光線、車輛距離與OCR模型本身效能都會影響到文字辨識準確度的重要因素。未來，我們將嘗試各種文字辨識方法，改善文字預處理的流程並提高執行速度，使這套自動化的貨櫃編號辨識系統更加實用。
