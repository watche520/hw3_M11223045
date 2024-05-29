import torch
import os
import yaml

from fastercnn.models import create_fasterrcnn_model
from fastercnn.utils.transforms import infer_transforms

# 設定執行裝置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 設定模型權重路徑
weights_path = os.path.join('fastercnn', 'weights', 'best_model.pth')

# 讀取模型權重
checkpoint = torch.load(weights_path, map_location=device)
num_classes = checkpoint['data']['NC']
classes_name = checkpoint['data']['CLASSES']
# 建立模型
model = create_fasterrcnn_model.return_fasterrcnn_resnet50_fpn_v2(num_classes=num_classes, coco_model=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

def inference_img(opencv_image) :
    image = infer_transforms(opencv_image)
    # Add batch dimension.
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image.to(device))
    # 把資料讀回 CPU
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    return outputs