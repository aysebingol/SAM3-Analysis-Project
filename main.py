import os
import cv2
from roboflow import Roboflow
from ultralytics import SAM

print("Yaprak Dataseti Testi")

rf = Roboflow(api_key="hPHyM5X03BAroeXFXlbT")
project = rf.workspace("workspace-bt4sl").project("segmentation-c5uow")
version = project.version(4)  
dataset = version.download("coco") 

image_dir = os.path.join(dataset.location, "train")
if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
    image_dir = os.path.join(dataset.location, "train")

output_dir = "sam3_yaprak_sonuclar"
os.makedirs(output_dir, exist_ok=True)

model_yolu = "/home/aysebingol/Downloads/sam3.pt"
model = SAM(model_yolu)

images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
total_images = len(images)
print(f"Toplam {total_images} gorsel bulundu. islem basliyor")

for idx, img_name in enumerate(images):
    img_path = os.path.join(image_dir, img_name)
    
    print(f"[{idx+1}/{total_images}] isleniyor: {img_name}")

    
    results = model.predict(
        source=img_path,
        device="cpu",      
        conf=0.25,         
        imgsz=1024,
        verbose=False     
    )

    if results and results[0].masks is not None:
        
        annotated = results[0].plot()
        
        
        save_path = os.path.join(output_dir, f"result_{img_name}")
        cv2.imwrite(save_path, annotated)
    else:
        print(f" {img_name} icin maske olusturulamadi.")

print(f"\n islem basariyla tamamlandi!")
print(f"sonuclari bu klasorde bulabilirsin: {os.path.abspath(output_dir)}")