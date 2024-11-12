import os
from ultralytics import YOLO

# Gerekli ortam değişkenini ayarla
os.environ["OMP_NUM_THREADS"] = '2'

# Modeli yükle (pretrained bir model veya yeni bir model)
#model = YOLO('yolov8n.yaml')  # Yeni bir model oluşturmak için YAML dosyası
# veya önceden eğitilmiş bir modeli yükleyin
model = YOLO('yolov8n.pt')  # Önerilen önceden eğitilmiş model

# Modeli eğit
model.train(data='C:/Users/baran/projeler/pycharmProject/finalOdev/data.yaml',
            epochs=50,  # Eğitimin kaç epoch süreceği
            imgsz=640,  # Görüntü boyutu
            batch=16,  # Batch boyutu
            device='cpu')  # Kullanılacak GPU cihazı
