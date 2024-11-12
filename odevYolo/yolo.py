import cv2
from ultralytics import YOLO

# YOLOv8 modelini yükle (nano model)
model = YOLO('yolov8n.pt')  # YOLOv8 nano modeli
model = YOLO('C:/Users/baran/runs/detect/train7/weights/best.pt')


# Web kamerasını başlat (kamera ID'si 0 ile genellikle ilk kamera açılır)
cap = cv2.VideoCapture(0)

# Web kamerası doğru bir şekilde açılmazsa hata verir
if not cap.isOpened():
    print("Kamera açılamadı")
    exit()

# Web kamerasından gelen veriyi sürekli işleme
while True:
    ret, frame = cap.read()  # Web kamerasından bir kare oku
    if not ret:
        print("Kare alınamadı, çıkış yapılıyor...")
        break

    # Her karede YOLOv8 ile tespit yap
    results = model(frame)

    # Tespit edilen nesneleri işle
    annotated_frame = results[0].plot()  # Sonuçları görselleştir (ilk sonuç için)

    # Web kamerasından gelen görüntüyü OpenCV ile göster
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
