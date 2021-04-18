import cv2
import numpy as np

# Yolo kütüphanesiin yüklenmesi
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") # YoloV3 hazır kütüphanesini programımıza kaydediyoruz
siniflar = []
with open("nesneler.names", "r") as f:
    siniflar = [satir.strip() for satir in f.readlines()] #nesne isimleri satırlarını okuyoruz
katman_isimleri = net.getLayerNames() # yolodan nesne analizi için katmanları alıyoruz
cikis_katmanlari = [katman_isimleri[i[0] - 1] for i in net.getUnconnectedOutLayers()]  #yolodan çıkış katmanlarını alıyoruz
colors = np.random.uniform(0, 0, size=(len(siniflar), 3)) #nesnelerin üstüne yazdırdığımız isimlerin hangi renkte olduğunu belirliyoruz

#print(katman_isimleri)
#print(cikis_katmanlari)
#print(colors)


# fotoğraf yükleme kısmı
img = cv2.imread("sea.jpg")  #fotoğrafı okuyor
img = cv2.resize(img, None, fx=0.4, fy=0.4) # boyutlarını ayarlıyor
height, width, channels = img.shape


# nesneleri tespit ettiğimiz yer
tespit = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(tespit)
cikislar = net.forward(cikis_katmanlari)

# Nesneleri ekrana yazdırma bilgisi 
class_ids = []   # sınıf isim bilgileri için oluşturduğumuz dizi
confidences = [] # %kaç benzediğini bulmak için oluşturuyoruz
boxes = []
for cikis in cikislar:
    for detection in cikis:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]                
        if confidence > 0.5:   #BURDA BENZERLİK 0.5 DEN BÜYÜKSE YANİ YARIDAN FAZLAYSA EKRANA NESNEYE GÖRE BİR DİKDÖRTGEN ÇİZDİRİYOR 
            # NESNE tespiti
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # dikdörtgenin çizilceği yerin koordinatları
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
yazi_tipi = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(siniflar[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), yazi_tipi, 3, color, 3)


cv2.imshow("Nesne Tespiti", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

