import cv2
import numpy as np
import os
import pandas as pd

def calculate_advanced_metrics(gt_mask, pred_mask):
    gt = (gt_mask > 0).astype(np.uint8)
    pred = (pred_mask > 0).astype(np.uint8)

    tp = np.logical_and(gt, pred).sum()  
    fp = np.logical_and(np.logical_not(gt), pred).sum()  
    fn = np.logical_and(gt, np.logical_not(pred)).sum() 
    
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    return iou, precision, recall, dice

gt_path = '/home/aysebingol/Downloads/sam3/real_masks' 
pred_path = '/home/aysebingol/Downloads/sam3/sam3_yaprak_sonuclar'

results = []
print("Gerçek metrikler hesaplanıyor...")

for filename in os.listdir(pred_path):
    if filename.startswith("result_"):
        gt_filename = filename.replace("result_", "")
        gt_file = os.path.join(gt_path, gt_filename)
        pred_file = os.path.join(pred_path, filename)

        if os.path.exists(gt_file):
            gt_img = cv2.imread(gt_file, 0) 
            pred_img = cv2.imread(pred_file, 0) 

            if gt_img is not None and pred_img is not None:
                if gt_img.shape != pred_img.shape:
                    pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
                
                iou, prec, rec, dice = calculate_advanced_metrics(gt_img, pred_img)
                results.append({
                    "Dosya": gt_filename, 
                    "IoU": iou, 
                    "Precision": prec, 
                    "Recall": rec,
                    "Dice_Score": dice
                })
        else:
            print(f" Eşleşen maske bulunamadı: {gt_filename}")

# Raporu Oluştur
if results:
    df = pd.DataFrame(results)
    df.to_csv("gercek_performans_raporu.csv", index=False)
    
    print("\n --- GERÇEK ANALİZ SONUÇLARI ---")
    print(f"Ortalama IoU: {df['IoU'].mean():.4f}")
    print(f"Ortalama Precision: {df['Precision'].mean():.4f}")
    print(f"Ortalama Recall: {df['Recall'].mean():.4f}")
    print(f"En Düşük IoU: {df['IoU'].min():.4f} (Zorlandığı resim)")
    print(f"\nSonuçlar 'gercek_performans_raporu.csv' dosyasına kaydedildi.")
else:
    print("Hata: Hiçbir dosya kıyaslanamadı! Klasör yollarını kontrol et.")