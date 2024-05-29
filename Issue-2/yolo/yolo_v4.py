import cv2
from yolo import darknet

network, class_names, class_colors = darknet.load_network(r'yolo/weights/yolov4-custom.cfg',
                                                          r'yolo/weights/obj.data',
                                                          r'yolo/weights/yolov4-custom_last.weights')

network_width = darknet.network_width(network)
network_height = darknet.network_height(network)

def image_detector(img):
    # 建立一張Darknet的空白圖片
    darknet_image = darknet.make_image(network_width, network_height, 3)
    # 將原始圖片轉為RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 依據網路的規格調整圖片尺寸 resize(被修改影像, (w, h), interpolation=cv2.INTER_LINEAR(插值方式))
    img_resized = cv2.resize(img_rgb, (network_width, network_height), interpolation=cv2.INTER_LINEAR)
    # 取得圖片的長寬，讓畫框時比例正常
    img_height, img_width, _ = img.shape
    height_ratio = img_height / network_height
    width_ratio = img_width / network_width
    # 將調整過大小的影像插入Darknet空白圖片
    darknet.copy_image_from_bytes(darknet_image, img_resized.tobytes())
    # 辨識圖片
    detections = darknet.detect_image(network, class_names, darknet_image)
    # 清除圖片
    darknet.free_image(darknet_image)
    return detections, height_ratio, width_ratio

def yolo_bbox_2_xyxy(bbox, height_ratio, width_ratio):
    x_min, y_min, x_max, y_max = darknet.bbox2points(bbox)
    x_min, y_min, x_max, y_max = int(x_min * width_ratio), int(y_min * height_ratio), int(x_max * width_ratio), int(y_max * height_ratio)
    # 在邊緣的避免出現負數
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = max(0, x_max)
    y_max = max(0, y_max)
    return x_min, y_min, x_max, y_max


if __name__ == '__main__':
    img = cv2.imread(r'yolo/image_0001.jpg')
    detections, height_ratio, width_ratio = image_detector(img)
    print(detections)
    for label, confidence, bbox in detections:
        x_min, y_min, x_max, y_max = yolo_bbox_2_xyxy(bbox, height_ratio, width_ratio)