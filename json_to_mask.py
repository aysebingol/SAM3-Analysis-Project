import json
import cv2
import numpy as np
import os

json_path = '/home/aysebingol/Downloads/sam3/segmentation-4/train/_annotations.coco.json'
img_dir = '/home/aysebingol/Downloads/sam3/segmentation-4/train'
output_dir = '/home/aysebingol/Downloads/sam3/real_masks'
os.makedirs(output_dir, exist_ok=True)

with open(json_path) as f:
    data = json.load(f)

for img_info in data['images']:
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
    for ann in data['annotations']:
        if ann['image_id'] == img_info['id']:
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((len(seg) // 2, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 255)
    cv2.imwrite(os.path.join(output_dir, img_info['file_name']), mask)
print("✅ Gerçek maskeler 'real_masks' klasörüne kaydedildi!")