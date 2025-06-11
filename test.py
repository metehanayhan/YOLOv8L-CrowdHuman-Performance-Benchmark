import time
import glob
import os
from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
 
    model = YOLO("final_ready_for_yolo.pt")  # veya yolov8n.pt

    val_results = model.val(
        data="data.yaml",  # data.yaml içinde val: /path/to/images/val
        imgsz=640,
        device=0,   # GPU:0, CPU:-1
        workers=4,
        batch=16
    )

    images_dir = "images/val"
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))

    if image_files:
        _ = model.predict(source=image_files[0], device=0, imgsz=640)

    start_time = time.time()
    for img_path in image_files:
        _ = model.predict(
            source=img_path,
            device=0,
            imgsz=640,
            conf=0.25,
            verbose=False
        )
    end_time = time.time()

    elapsed = end_time - start_time
    num_images = len(image_files)
    if num_images > 0 and elapsed > 0:
        fps = num_images / elapsed
        latency = elapsed / num_images
    else:
        fps = 0
        latency = 0

    print("\n=== INFERENCE SPEED ===")
    print(f"Görüntü sayısı: {num_images}")
    print(f"Toplam süre: {elapsed:.2f} s")
    print(f"Ortalama FPS: {fps:.2f}")
    print(f"Ortalama Latency: {latency:.3f} s/görüntü (~{latency*1000:.2f} ms)")
    print("\nTüm işlemler tamamlandı.")

if __name__ == "__main__":
    freeze_support() 
    main()
