import cv2
import numpy as np
import requests
from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')


# URL'den profil resmini indirip inceleyen fonksiyon
def analyze_profile_image(profile_url):
    try:
        # URL'den profil resmini çek
        profile_image_url = f"{profile_url}/picture?type=large"
        response = requests.get(profile_image_url, stream=True)

        if response.status_code == 200:
            img_data = response.content
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                return "Resim yüklenemedi"

            # YOLO ile insan olup olmadığını kontrol et
            results = model(img)
            confidence_threshold = 0.5  # Minimum güven eşiği

            # İnsan tespit etme
            for result in results:
                class_ids = result.boxes.cls
                confidences = result.boxes.conf
                for class_id, confidence in zip(class_ids, confidences):
                    if model.names[int(class_id)] == 'person' and confidence > confidence_threshold:
                        return f"İNSAN (Güven: {confidence:.2f})"
            return "BASKA BİR VARLIK"
        else:
            return "Resim indirilemedi"
    except Exception as e:
        return f"Hata: {str(e)}"


# Facebook profil URL'lerini listele
profile_urls = [
    "https://cdn1.ntv.com.tr/gorsel/qH05QRSOgEG_NIDTii5D2Q.jpg?width=660&mode=both&scale=both",
    "https://cdn1.ntv.com.tr/gorsel/CckH_t_YcU2zOMXnpp8U_g.jpg?width=660&height=495&mode=both&scale=both",
    "https://cdn.ntvspor.net/28f83cc79dfa452da774ff654f761d6c.jpg?mode=crop&w=940&h=626",
    "https://cdn.ntvspor.net/047f56659f2748cf92bd636a135dd39d.jpg?w=660",
    "https://cdn.ntvspor.net/c3f41736a032476a9d9add4bb8790730.jpg?w=660",
    "https://cdn.ntvspor.net/c0619f36e26f430d9fece89c4defe2c0.jpg?mode=crop&w=940&h=626",
    "https://cdn.ntvspor.net/db5ee47cfe55410ab3b5186443333d8c.jpg?mode=crop&w=940&h=626",
    "https://cdn.ntvspor.net/18e65fa3d7594c318289e1520cc6800d.jpg?mode=crop&w=940&h=626",
    "https://cdn.ntvspor.net/eccb91fc3df34887979a8491fe4d77e2.jpg?mode=crop&w=940&h=626",
    "http://cdn.ntvspor.net/c5a11cdab4c846938f91d673cdcf5857.jpg?mode=crop&w=940&h=626",
    ]


# Her profil için analiz yap
for url in profile_urls:
    result = analyze_profile_image(url)
    print(f"{url}: {result}")
