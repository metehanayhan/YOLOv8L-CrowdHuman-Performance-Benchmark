import os
import json
from PIL import Image

ODGT_FILE = "/home/pc1/Desktop/yoloo/annotation_val.odgt"

IMAGES_VAL_DIR = "/home/pc1/Desktop/yoloo/images/val"

LABELS_VAL_DIR = "/home/pc1/Desktop/yoloo/labels/val"

CLASS_ID = 0

os.makedirs(LABELS_VAL_DIR, exist_ok=True)

with open(ODGT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line:
        continue
    
    # Her satır JSON: {"ID": "...", "gtboxes": [...], ...}
    data = json.loads(line)
    
    image_id = data["ID"]
    image_name = image_id + ".jpg"
    
    image_path = os.path.join(IMAGES_VAL_DIR, image_name)
    
    if not os.path.exists(image_path):
        print(f"[Uyarı] Resim yok: {image_path}")
        continue
    
    with Image.open(image_path) as im:
        img_width, img_height = im.size
    
    # Bu resmin YOLO formatlı etiket satırlarını toplayacağımız liste
    yolo_lines = []
    
    # gtboxes içindeki her nesneyi oku
    for obj in data.get("gtboxes", []):
        # Sadece vbox kullanalım
        if "vbox" not in obj:
            continue
        
        vbox = obj["vbox"]  # [x, y, w, h]
        x, y, w, h = vbox
        
        # YOLO formatına dönüştürelim.. (normalize)
        x_center = (2*x + w) / 2.0 / img_width   # (x + x + w)/2
        y_center = (2*y + h) / 2.0 / img_height
        w_norm   = w / img_width
        h_norm   = h / img_height
        
        # Değerler 0-1 arası mı? 1'i aşıyorsa corrupt olabilir, atlıyoruz.
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                0 <= w_norm <= 1 and 0 <= h_norm <= 1):
            print(f"[Bilgi] Out-of-bounds bbox, atlanıyor: {image_name} => {vbox}")
            continue
        
        # Tek sınıf ID = 0
        yolo_line = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        yolo_lines.append(yolo_line)
    
    # Artık toplanan satırları .txt dosyasına yazalım
    txt_name = image_id + ".txt"
    txt_path = os.path.join(LABELS_VAL_DIR, txt_name)
    
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write("\n".join(yolo_lines) + "\n" if yolo_lines else "")
    
    print(f"[INFO] {image_name} için {len(yolo_lines)} bbox etiketi yazıldı -> {txt_path}")
