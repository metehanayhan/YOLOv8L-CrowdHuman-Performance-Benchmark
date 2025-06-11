[TR]
# YOLOv8-L Performans Analizi ve Model Karşılaştırması (CrowdHuman Dataset)

Bu proje, Engineer Hub Programı kapsamında gerçekleştirilmiştir. Projenin amacı, kalabalık insan içeren görüntülerde nesne tespiti için YOLOv8-L modelinin doğruluk ve hız performansını ölçmek, ayrıca diğer YOLO varyantlarıyla karşılaştırmalı analiz yapmaktır.

---

## 📘 İçindekiler

1. [Giriş](#1-giriş)  
2. [Metodoloji](#2-metodoloji)  
3. [Metrik Tanımları](#3-metrik-tanımları)  
4. [Sonuçlar ve Değerlendirme](#4-sonuçlar-ve-değerlendirme)  
5. [GPU Kullanımı ve Güç Ölçümü](#5-gpu-kullanımı-ve-güç-ölçümü)  
6. [Model Karşılaştırma Tablosu](#6-model-karşılaştırma-tablosu)  
7. [Genel Değerlendirme ve Öneriler](#7-genel-değerlendirme-ve-öneriler)  
8. [Kaynakça](#8-kaynakça)

---

## 1. Giriş

Bu çalışmada, YOLOv8-L (Large) modelinin insan tespiti odaklı bir veri seti (CrowdHuman) üzerinde doğruluk (precision, recall, mAP) ve hız (FPS, latency) performansları analiz edilmiştir. Toplamda 4370 görüntü ve 127.639 insan etiketi içeren veri seti, kalabalık sahnelerde gerçek zamanlı nesne tespiti problemleri için kullanılmaktadır.

---

## 2. Metodoloji

### 2.1 Model
- Kullanılan model: Ultralytics YOLOv8-L (önceden eğitilmiş)
- Model boyutu: 43.668.288 parametre
- Hesaplama yükü: 165.2 GFLOPs

### 2.2 Veri Seti
- Görüntü sayısı: 4370 (val set)
- Toplam nesne etiketi: 127.639 kişi (class: person)
- Klasör yapısı:
  - `images/val`
  - `labels/val` (YOLO formatında)

### 2.3 Değerlendirme Yöntemi
1. **Doğruluk:** `model.val()` komutuyla precision, recall ve mAP hesaplandı.  
2. **FPS (hız):** 4370 görüntü tek tek `model.predict()` ile işlendi. `time.time()` ile süresi ölçüldü.  
3. **GPU Ölçümü:** `pynvml` ile inference sırasında VRAM ve güç tüketimi hesaplandı.

---

## 3. Metrik Tanımları

| Metrik | Açıklama |
|--------|----------|
| **Precision (P)** | Doğru pozitiflerin tüm pozitif tahminlere oranı |
| **Recall (R)** | Doğru pozitiflerin tüm gerçek pozitiflere oranı |
| **mAP@0.5** | IoU=0.5 eşik değerinde ortalama hassasiyet |
| **mAP@0.5:0.95** | IoU=0.5'ten 0.95'e kadar değişen eşiklerde ortalama hassasiyet |
| **FPS** | Saniyede işlenen görüntü sayısı |
| **Latency** | Görüntü başına düşen ortalama işlem süresi |

---

## 4. Sonuçlar ve Değerlendirme

### 4.1 Val (Doğruluk) Metrikleri

| Sınıf | Görsel | Nesne | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|--------|-------|-----------|--------|---------|---------------|
| all / person | 4370 | 127639 | 0.752 | 0.517 | 0.629 | 0.408 |

- **Precision:** %75.2 — düşük yanlış pozitif oranı
- **Recall:** %51.7 — kalabalık ortamlarda makul
- **mAP@0.5:** %62.9 — orta-üst düzey doğruluk
- **mAP@0.5:0.95:** %40.8 — sıkı eşiklerde başarı

### 4.2 Inference Hızı

- Toplam süre: **86.26 saniye**
- Ortalama FPS: **50.66**
- Ortalama latency: **~19.74 ms/görüntü**

> Bu değerler RTX 3070 GPU üzerinde elde edilmiştir ve gerçek zamanlı uygulamalar için oldukça yeterlidir.

---

## 5. GPU Kullanımı ve Güç Ölçümü

`pynvml` ile yapılan ölçümlere göre:

- **En Yüksek VRAM Kullanımı:** 1247.4 MB
- **Ortalama Güç Tüketimi:** 124.6 W
- **Maksimum Güç Tüketimi:** 139.2 W
- **Toplam Inference Süresi:** 93.65 s
- **Ortalama FPS:** 46.66

> 1.2 GB VRAM kullanımı, 8 GB kapasiteli RTX 3070 için oldukça verimli bir seviyedir.

---

## 6. Model Karşılaştırma Tablosu

| Model     | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | FPS   | Latency (ms) |
|-----------|-----------|--------|---------|---------------|--------|---------------|
| YOLOv8n   | 0.674     | 0.445  | 0.529   | 0.276         | 107.31 | 9.32 |
| YOLOv8m   | 0.693     | 0.501  | 0.575   | 0.311         | 66.4   | 15.06 |
| YOLOv8l   | 0.687     | 0.507  | 0.577   | 0.316         | 50.44  | 19.83 |
| YOLOv8x   | 0.690     | 0.514  | 0.584   | 0.321         | 38.69  | 25.85 |
| YOLOv10x  | 0.687     | 0.495  | 0.569   | 0.315         | 44.29  | 22.58 |
| YOLOv11x  | 0.693     | 0.508  | 0.583   | 0.323         | 40.3   | 24.81 |
| YOLOv11n  | 0.667     | 0.438  | 0.523   | 0.273         | 99.94  | 10.01 |
| RT-DETR-x | 0.656     | 0.481  | 0.535   | 0.267         | 25.1   | 39.84 |

---

## 7. Genel Değerlendirme ve Öneriler

- **Performans:** YOLOv8-L modeli, %75 precision ve %63 mAP ile güçlü bir doğruluk seviyesi sunar.
- **Hız:** 50 FPS ile birçok gerçek zamanlı uygulama için uygundur.
- **Kompromis:** Model boyutu büyüdükçe doğruluk artar ama FPS düşer.  
- **Seçim:**  
  - Gerçek zamanlılık önemliyse: `YOLOv8n / YOLOv11n`  
  - Doğruluk öncelikliyse: `YOLOv11x / YOLOv8x`  

---

## 8. Kaynakça

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- https://www.stereolabs.com/blog/performance-of-yolo-v5-v7-and-v8
- https://github.com/autogyro/yolo-V8
- https://github.com/ultralytics/ultralytics/issues/2690

---

> Bu proje, Aselsan Konya Laboratuvarı'ndaki RTX 3070 GPU ile gerçekleştirilmiş, donanım seviyesinde güç ve hız analizleriyle desteklenmiştir.  

[EN]
# YOLOv8-L Performance Analysis and Model Comparison (CrowdHuman Dataset)

This project was carried out as part of the Engineer Hub Program. The aim is to evaluate the accuracy and speed performance of the YOLOv8-L model in detecting humans in highly crowded scenes, and to compare it with other YOLO variants.

---

## 📘 Table of Contents

1. [Introduction](#1-introduction)  
2. [Methodology](#2-methodology)  
3. [Metric Definitions](#3-metric-definitions)  
4. [Results and Evaluation](#4-results-and-evaluation)  
5. [GPU Usage and Power Measurement](#5-gpu-usage-and-power-measurement)  
6. [Model Comparison Table](#6-model-comparison-table)  
7. [General Evaluation and Recommendations](#7-general-evaluation-and-recommendations)  
8. [References](#8-references)

---

## 1. Introduction

This study analyzes the performance of the YOLOv8-L (Large) model in terms of accuracy (precision, recall, mAP) and speed (FPS, latency) on a human detection-focused dataset, CrowdHuman. The dataset includes 4370 images and a total of 127,639 person labels, making it suitable for real-time object detection scenarios in crowded environments.

---

## 2. Methodology

### 2.1 Model
- Model used: Ultralytics YOLOv8-L (pretrained)
- Model size: 43,668,288 parameters
- Computational load: 165.2 GFLOPs

### 2.2 Dataset
- Number of images: 4370 (validation set)
- Total object instances: 127,639 (class: person)
- Folder structure:
  - `images/val`
  - `labels/val` (YOLO format)

### 2.3 Evaluation Method
1. **Accuracy:** Calculated using `model.val()` to obtain precision, recall, and mAP.
2. **Inference Speed (FPS):** Each image processed using `model.predict()` and time measured with `time.time()`.
3. **GPU Metrics:** VRAM and power usage collected using `pynvml` during inference.

---

## 3. Metric Definitions

| Metric | Description |
|--------|-------------|
| **Precision (P)** | Ratio of true positive predictions to total positive predictions |
| **Recall (R)** | Ratio of true positives to actual positives |
| **mAP@0.5** | Mean Average Precision at 0.5 IoU |
| **mAP@0.5:0.95** | Average Precision over IoU thresholds from 0.5 to 0.95 |
| **FPS** | Frames processed per second |
| **Latency** | Average processing time per image |

---

## 4. Results and Evaluation

### 4.1 Validation Metrics

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|--------|-----------|-----------|--------|---------|---------------|
| all / person | 4370 | 127639 | 0.752 | 0.517 | 0.629 | 0.408 |

- **Precision:** 75.2% — low false positives
- **Recall:** 51.7% — reasonable in crowded scenes
- **mAP@0.5:** 62.9% — moderate to high accuracy
- **mAP@0.5:0.95:** 40.8% — strict IoU thresholds

### 4.2 Inference Speed

- Total time: **86.26 seconds**
- Average FPS: **50.66**
- Average latency: **~19.74 ms/image**

> These values were obtained using an NVIDIA RTX 3070 GPU and are suitable for real-time applications.

---

## 5. GPU Usage and Power Measurement

Measured using `pynvml`:

- **Max VRAM usage:** 1247.4 MB
- **Average power usage:** 124.6 W
- **Max power draw:** 139.2 W
- **Total inference time:** 93.65 seconds
- **Average FPS:** 46.66

> 1.2 GB VRAM usage is efficient for an 8 GB RTX 3070 GPU.

---

## 6. Model Comparison Table

| Model     | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | FPS   | Latency (ms) |
|-----------|-----------|--------|---------|---------------|--------|---------------|
| YOLOv8n   | 0.674     | 0.445  | 0.529   | 0.276         | 107.31 | 9.32 |
| YOLOv8m   | 0.693     | 0.501  | 0.575   | 0.311         | 66.4   | 15.06 |
| YOLOv8l   | 0.687     | 0.507  | 0.577   | 0.316         | 50.44  | 19.83 |
| YOLOv8x   | 0.690     | 0.514  | 0.584   | 0.321         | 38.69  | 25.85 |
| YOLOv10x  | 0.687     | 0.495  | 0.569   | 0.315         | 44.29  | 22.58 |
| YOLOv11x  | 0.693     | 0.508  | 0.583   | 0.323         | 40.3   | 24.81 |
| YOLOv11n  | 0.667     | 0.438  | 0.523   | 0.273         | 99.94  | 10.01 |
| RT-DETR-x | 0.656     | 0.481  | 0.535   | 0.267         | 25.1   | 39.84 |

---

## 7. General Evaluation and Recommendations

- **Performance:** YOLOv8-L delivers strong precision and mAP results.
- **Speed:** With 50+ FPS, it suits real-time applications.
- **Trade-off:** Larger models yield better accuracy but lower FPS.
- **Recommendation:**
  - For real-time use: `YOLOv8n` or `YOLOv11n`
  - For maximum accuracy: `YOLOv8x` or `YOLOv11x`

---

## 8. References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- https://www.stereolabs.com/blog/performance-of-yolo-v5-v7-and-v8
- https://github.com/autogyro/yolo-V8
- https://github.com/ultralytics/ultralytics/issues/2690

---

> This project was conducted on RTX 3070 GPUs in the Aselsan Konya Laboratory environment with detailed power and memory performance measurements.