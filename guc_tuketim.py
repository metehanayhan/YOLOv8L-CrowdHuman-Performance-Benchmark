import time
import glob
import os

from pynvml import (
    nvmlInit, nvmlShutdown,
    nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage,
)
from ultralytics import YOLO


nvmlInit()
device_count = nvmlDeviceGetCount()
handle = nvmlDeviceGetHandleByIndex(0)

model = YOLO("yolov8l.pt")  # ya da yolov8n.pt

images_dir = "/home/pc1/Desktop/yoloo/images/val"
image_files = glob.glob(os.path.join(images_dir, "*.jpg"))

# (Opsiyonel) GPU Isınma 
if image_files:
    model.predict(source=image_files[0], device=0)

num_images = len(image_files)
start_time = time.time()

# Maks ve toplu verileri tutmak için değişkenler
max_mem = 0          # MB cinsinden en yüksek bellek kullanımı
power_readings = []  # Anlık güç değerlerini toplayacağız (Watt)


# INFERENCE + ÖLÇÜM
for img_path in image_files:
    # Inference
    model.predict(source=img_path, device=0)
    
    # Anlık bellek bilgisi
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    used_mb = mem_info.used / (1024**2)  # MB
    if used_mb > max_mem:
        max_mem = used_mb
    
    # Anlık güç tüketimi (milliwatt → watt)
    power_mw = nvmlDeviceGetPowerUsage(handle)
    power_w = power_mw / 1000.0
    power_readings.append(power_w)

end_time = time.time()
elapsed = end_time - start_time


if num_images > 0:
    fps = num_images / elapsed
    avg_power = sum(power_readings) / len(power_readings) if power_readings else 0
    max_power = max(power_readings) if power_readings else 0
else:
    fps = 0
    avg_power = 0
    max_power = 0

print(f"\n=== GPU ÖLÇÜMLERİ ===")
print(f"Toplam Görüntü Sayısı: {num_images}")
print(f"Süre (sn): {elapsed:.2f}")
print(f"FPS (ortalama): {fps:.2f}\n")

print(f"En Yüksek GPU Bellek Kullanımı: {max_mem:.1f} MB")
print(f"Ortalama Güç Tüketimi: {avg_power:.1f} W")
print(f"Maksimum Güç Tüketimi: {max_power:.1f} W")

nvmlShutdown()
