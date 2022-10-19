
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
import time


######################################### ÇİZİMİN ALANINI BELİRLEME ################################################

class Point: # kose noktalari icin bir sinif olustur
    def __init__(self, x, y):
        self.x = x # x ve y koordinatlarini tut
        self.y = y

    @staticmethod # statik bir oklid uzakligi formulu 
    def euclidianDistance(x1, x2, y1, y2):
        return np.sqrt( np.square(x1 - x2) + np.square( y1 - y2) )
    
    def distance(self, other): # iki nokta arasindaki uzakligi statik oklid uzakligi vasitasiyla bul
        return Point.euclidianDistance(self.x, other.x, self.y, other.y)

class Polygon: # cokgen sinifi olustur
    def __init__(self):
        self.points = [] # cokgenin noktalarini tut

    def addPoint(self, p1):
        self.points.append(p1) # yeni noktalari ekle

    def addPointList(self, points):
        for point in points:
            self.addPoint(point) # nokta listesi ekle

    def getArea(self): # alan algoritmasi
        area = 0.0 # bir akumulator degisken olustur
        for idx in range(len(self.points[1:])): # her nokta icin delta_x * sum(y1,y2) / 2 formulunu alana ekle
            area += (self.points[idx+1].x - self.points[idx].x) * (self.points[idx+1].y + self.points[idx].y) * 0.5
        area += (self.points[0].x - self.points[-1].x) * (self.points[0].y + self.points[-1].y) * 0.5 
        # ilk nokta ile son nokta arsinda ayni islemi yap
        return abs(area) # alanin mutlak degerini dondur

poly1 = Polygon()

pp1 = Point(215,50)
pp2 = Point(160,250)
pp3 = Point(500,250)
pp4 = Point(415,50)

poly1.addPointList([pp1,pp2,pp3,pp4])



#############################################     1.BÖLÜM SOSYALMESAFE KONFİGÜRASYON AYARLARI    ##################################################################

# YOLO dizinine giden temel yol
MODEL_PATH = "yolov3"
MIN_CONF = 0.5   #Zayıf algılamaları filtrelemek içi Güven değeri en az 0.5 olan nesneleri alacağız.
NMS_THRESH = 0.4 #NMS (Non Maximum Suppression)=
                  #Birbiriyle örtüşen birçok varlık arasından (örneğin, sınırlayıcı kutular) seçmek için kullanılan bir algoritma sınıfıdır. 

# boolean, NVIDIA CUDA GPU'nun kullanılması gerekip gerekmediğini gösterir.
USE_GPU = True
# iki kişinin algılanması durumunda minimum güvenli mesafeyi (piksel olarak) tanımlıyoruz.
#MIN_DISTANCE = 45



#TENSOR=verinin tutulması için kullanılan bir yapıdır. Matris gibi



###############################################    2. BÖLÜM DETECTİON İŞLEMİ     ##################################################################################

def detect_people(frame, net, ln, personIdx=0):
	# çerçevenin boyutlarını alıyoruz.
	(H, W) = frame.shape[:2]
	results = []
        #BLOB= uzamsal boyutlara (genişlik ve yükseklik), derinliğe (kanal sayısı) sahip bir görüntü topluluğudur.
	#Kısaca ortalama çıkarma ve bazı değerlere göre ölçeklendirmedir.
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)
	
	# algılanan sınırlayıcı kutular, merkezler ve güven değerlerinin tutulduğu listeleri tanımlıyoruz.
	boxes = []
	centroids = []
	confidences = []
	toplam=0
	sayac=0
	

        # katman çıktılarının her biri üzerinde döngü
	for output in layerOutputs:
		# algılamaların her biri üzerinde döngü
		for detection in output:
			
			scores = detection[5:]       # ilk 5 değer boundingBox ile ilgilidir
			classID = np.argmax(scores)  # Enyüksek skoru ClassID ye ata
			confidence = scores[classID] # Güven skorlarının tutulduğu değişken
			
			# Nesne varsa ve Güven 0.5(MIN_CONF) ten büyük ise
			if classID == personIdx and confidence > MIN_CONF:

				# YOLO'yu akılda tutarak sınırlayıcı kutu koordinatlarını ölçeklendireceğiz.
				# merkez (x, y) koordinatlarını ve sınırlayıcı kutunun genişliğini ve yüksekliğini belirliyoruz.
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# merkez (x, y) koordinatlarını kullanarak
				# tepeyi (sınırlayıcı kutunun sol üst köşesi) elde ediyoruz.
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				mesafe=int(height)-int(width)#İki kişi arasındaki minimum mesafe
				toplam=mesafe+toplam
				sayac+=1
				ort=toplam/sayac
				
				
				# sınırlayıcı kutu koordinatlarını, merkez ve güven skorunu güncelliyoruz
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY,ort))
				confidences.append(float(confidence))
				
				
				
				


        # NMS ile en güvenilir sınırlayıcı kutuları alıyoruz
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
	
	# en az bir algılama var ise
	if len(idxs) > 0:
		# tuttuğumuz dizinler üzerinde döngü
		for i in idxs.flatten():
                        
			# sınırlayıcı kutu koordinatlarını çıkar
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
						
			# güven değeri, merkez ve sınırlayıcı kutu koordinatlarını alıp
			# Kişi sonuç listemizi güncelliyoruz
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
						
			results.append(r)
			
		# sonuç listesini döndür
	return results








############################    3.BÖLÜM SOSYAL MESAFE VE ALAN BELİRLEME İŞLEMLERİ     ################################################################################

# argüman ayrıştırıcısı ile GİRİŞ ve ÇIKIŞ işlemlerini belirliyoruz.
ap = argparse.ArgumentParser()
ap.add_argument('--webcam', help="True/False", default=False)
ap.add_argument("-i", "--input", type=str, default=False,help="(isteğe bağlı) giriş video dosyasının yolu" )
ap.add_argument("-o", "--output", type=str, default="",help="(isteğe bağlı) çıktı video dosyasının yolu" )
ap.add_argument("-d", "--display", type=int, default=1,help="çıktı çerçevesinin görüntülenip görüntülenmeyeceği" )
args = vars(ap.parse_args())



# YOLO modelimizin eğitim aldığı COCO sınıfı etiketlerini yüklüyoruz ve yolunu (PATH) belirliyoruz.
labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# YOLO ağırlıklarına ve model konfigürasyonuna giden yolları belirliyoruz.
weightsPath = os.path.sep.join([MODEL_PATH, "yolov3.weights"]) 
configPath = os.path.sep.join([MODEL_PATH, "yolov3.cfg"])

# COCO veri setinde eğitilmiş YOLO nesne dedektörümüzü yükleyin (COCO veri setinde 80 sınıf varken, biz sadece 1 sınıf kullanacağız.)
print("[BİLGİ] YOLO diskten yükleniyor...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# GPU kullanıp kullanmayacağımızı kontrol edin
if USE_GPU==True:
        # CUDA'yı tercih edilen arka uç ve hedef olarak ayarla (CUDA: Nvidia'nın kendi platformundaki GPU'ları kullanır)
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# YOLO'dan sadece ihtiyacımız olan "çıktı" katman adlarını belirliyoruz.
#Modelimizin içindeki tüm katmanları alıp ln değişkenine atıyoruz.
ln = net.getLayerNames()
#Çektiğimiz layerlerden çıktı katmanı olanları alıyoruz
#çıktı katmanlarını ln değişkeninin içinde saklarız
#YOLO modelinin içinde 254 tane katman var. Biz yolo ile başlayan katmanları alacağız
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]



# video işlemleri
print("[BİLGİ] video akışına erişiyor...")
# varsa giriş videosunu açın, aksi takdirde web kamerası açılacaktır
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
#vs = cv2.VideoCapture("videos/AVM.mp4")
writer = None
#Videonun başlama zamanı
start = time.time()


#Videodaki toplam kare sayısı
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[BİLGİ] Videoda toplam {} kare var".format(total))

# video okunmaması durumunda ekrana çıkacak hata mesajları
except:
	print("[BİLGİ] Videodaki kare sayısı belirlenemedi")
	total = -1

area_1=[(215,50),(160,250),(500,250),(415,50)]
speed=0
#fps değeri için
frame_id=0
i=0
# video akışındaki kareler üzerinde döngü
while True:
         # giriş videosundan sonraki kareyi oku
        (grabbed, frame) = vs.read()
        #fps değerini öğrenmek için
        frame_id+=1
        
        # okunacak  kare yoksa, o zaman sona ulaştık
        if not grabbed:
                break
        
        speed+=1
        if speed % 3 != 0:
                continue
        
        count=0
        t_count=0
        
        # çerçeveyi yeniden boyutlandırın ve ardından içindeki insanları (ve yalnızca insanları) tespit ediyoruz.
        frame = imutils.resize(frame, width=800,height=600)#480p çözünürlük
        results = detect_people(frame, net, ln,personIdx=LABELS.index("person"))
        
        # minimum sosyal mesafeyi ihlal eden dizin kümesini başlat
        violate = set()


################# ALAN BELİRLEME VE SADECE KİŞİLERİ ALMA ###############################
        cv2.polylines(frame,[np.array(area_1,np.int32)],True,(255,0,0),2)  
        labels_id=LABELS[LABELS.index("person")]
#########################################################################################     

           # sonuçlar üzerinde döngü
        for (i, (prob, bbox, centroid)) in enumerate(results):
                # sınırlayıcı kutuyu ve merkez koordinatlarını çıkar, ardından açıklamanın rengini başlat
                (startX, startY, endX, endY) = bbox
                (cX, cY,Min_Mesafe) = centroid
                color = (0, 255, 0)
            
                t_count+=1
               
                
                
                # dizin çifti ihlal kümesinde mevcutsa, rengi güncelleyin
                if i in violate:
                        color = (0, 0, 255)
                        
        
                if labels_id in ["person"]:
            
                        result_area=cv2.pointPolygonTest(np.array(area_1,np.int32),(int(cX),int(cY)),False)
                
            
                        if result_area>=0:
                    
                            count+=1
                   
                        # En az iki kişinin algılanması gerekir (gerekli ikili mesafe haritalarımızı hesaplamak için)
                            if len(results) >= 2:
                
                                 # Tüm merkez çiftleri arasında ağırlık merkezlerini çıkar ve Öklid mesafelerini hesapla
                                centroids = np.array([r[2] for r in results])
                                D = dist.cdist(centroids, centroids, metric="euclidean")
                
                                # mesafe matrisinin üst üçgeni üzerinde döngü
                                for i in range(0, D.shape[0]):
                                    for j in range(i + 1, D.shape[1]):

                                    # herhangi iki merkez çifti arasındaki mesafenin daha az olup olmadığını kontrol ediyoruz.
                                        if D[i, j] < Min_Mesafe: #Minimum mesafe

                                        # Merkez çiftlerinin indeksleri ile mesafe ihlali yapanları güncelle
                                            violate.add(i)
                                            violate.add(j)
                                            print("Distance : {}".format(str(round(Min_Mesafe)))+" "+"Piksel")

                                #Kişinin çevresine bir sınırlayıcı kutu ve kişinin merkez koordinatlarını çiziyoruz
                                #cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                                cv2.circle(frame, (cX, cY), 2, color, 2)
                
        #çıktı çerçevesine toplam sosyal mesafe ihlali sayısını çiziyoruz
        #text = "Sosyal Mesafe ihlalleri: {}".format(len(violate))
        #cv2.putText(frame, text, (8, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)
                                
        #MESAFE BİLGİSİNİ YAZDIRMAK İÇİN KULLANILDI
        #text3= "mesafe: {}".format(str(Min_Mesafe))
        #cv2.putText(frame, text3, (8, frame.shape[0] - 110),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 2)
        
        #text3= "Alandaki TOPLAM kişi Sayisi: {}".format(str(t_count))
        #cv2.putText(frame, text3, (8, frame.shape[0] - 90),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 2)
        
        #text1= "Alanin Piksel Sayisi: {}".format(str(poly1.getArea()))
        #cv2.putText(frame, text1, (8, frame.shape[0] - 70),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
        
        #text2= "Alanin Metrekaresi: {}".format(str(round(poly1.getArea()*0.000265,1)))
        #cv2.putText(frame, text2, (8, frame.shape[0] - 50),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
        
        #text3= "Alandaki Kisi Sayisi: {}".format(str(count))
        #cv2.putText(frame, text3, (8, frame.shape[0] - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 2)
               
        #text4= "Alanda Bulunmasi Gereken Max Kisi: {}".format(str(round((poly1.getArea()*0.000265),0)))
        #cv2.putText(frame, text4, (8, frame.shape[0] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 2)

        print("---------------------------------")
        print("---------------------------------")
        #Ekran görüntüsünüü kare kare alır
        cv2.imwrite('v3'+str(i)+'.jpg',frame)
        i+=1
        
        #TOPLAM KARE
        value_fps=vs.get(cv2.CAP_PROP_FPS)
        print("[TOPLAM FPS] {}".format(value_fps))
        
        #FPS HIZI
        elapsed_time = time.time() - start
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
        print("[FPS hızı] {}".format(round(fps,2)))

        # frame başına geçen saniye
        end=time.time()#Sonraki kare için geçen son saniye
        second = (end - start)
        print("[SÜRE] {:.4f} saniye".format(round(second,2)))

        
                
        
       
        #print ("Toplam Piksel (Alan): ",poly1.getArea())
        #print ("Metrekare (Alan): ",poly1.getArea()*0.0002645833)
        #print ("Metrekareye düşen kişi sayısı: ",poly1.getArea()*0.0002645833/count)
        
        # çıktı çerçevesinin ekranda gösterilip gösterilmeyeceğini kontrol edin
        if args["display"] > 0:
                # çıktı çerçevesini göster
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                
                 # 'q' tuşuna basılırsa döngüden çık
                if key == ord("q"):
                        break
                
        # bir çıkış video dosyası yolu sağlanmışsa ve video yazıcısı çalışmıyorsa
        if args["output"] != "" and writer is None:
                # video yazıcısını başlat
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 25,(frame.shape[1], frame.shape[0]), True)
        
                
                
        # video yazıcısı yoksa, çerçeveyi çıktıya yazın
        if writer is not None:
                writer.write(frame)
      
