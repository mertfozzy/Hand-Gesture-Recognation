# -*- coding: utf-8 -*-
"""
@author: mertf

"""

import cv2
import numpy as np
import math

vid = cv2.VideoCapture(0,cv2.CAP_DSHOW) # webcamden görüntü aldık

while (1) :
    try:
        ret, frame = vid.read()
        frame=cv2.flip(frame,1)

        kernel = np.ones((3,3), np.uint8) # kernel tanımlama sebebi morfolojik işlemler için

        # burada elin geleceği konumu yaratıyoruz :
        region_of_interest = frame[100:300,100:300]
        cv2.rectangle(frame, (100,100), (300,300),(0,0,255), 0)

        # bgr - hsv modu dönüşümü (belli alandaki rengi diğerlerinden ayırmak için)
        hsv = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2HSV) # el rengini ayırt edecek cilt rengi aralığını vereceğiz :
        lower_skin = np.array([0,20,70],np.uint8) # açık ten rengi alt limiti
        upper_skin = np.array([20,255,255],np.uint8) # koyu ten rengi üst limiti

        # mask oluşturuyoruz (mask yukardan ayırt ettiğimiz hsv modlu rengi ayırıyor)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.dilate(mask,kernel,iterations=4) # başta verdiğimiz 1lerden oluşan kernel değeri siyah noktaları beyazlaştıracak
        mask = cv2.GaussianBlur(mask, (5,5), 100) # blur efekti katacağız ve görüntüdeki kirlilik azalacak

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # countour noktalarını bulduk -default
        contoursBest = max(contours, key = lambda x: cv2.contourArea(x)) # countourlardan en yüksek oranlarını bulacak -default

        # counterlara daha iyi yaklaşım sağlamamız için matematiksel ifadeler hazırladık :
        epsilon = 0.0005*cv2.arcLength(contoursBest, True) # deneysel olarak denenmiş sayılar
        approx = cv2.approxPolyDP(contoursBest, epsilon, True)

        # el çevresine dış bükey örtü oluşturacağız :
        hull = cv2.convexHull(contoursBest) # koordinat elde ettik

        # koordinatlardan oluşacak şeklin alanını hesaplıyoruz :
        areaHull = cv2.contourArea(hull) # hull'un alanını oluşturduk
        areaCountour = cv2.contourArea(contoursBest) # elin alanını oluşturduk
        areaRatio = ((areaHull - areaCountour)/areaCountour)*100 # örtünün içinde elin olmadığı alanları hesapladık

        # dış-bükey kusurlarını bulacağız
        hull = cv2.convexHull(approx, returnPoints = False)
        defects = cv2.convexityDefects(approx, hull) # dış-bükey kusurlarını bulduk

        l = 0 # toplam kusur sayısı 0 verdik

        # kusur değerlerini değişkenlere atayacağız :
        for i in range (defects.shape[0]):
            s,e,f,d = defects[i,0] # başlangıç bitiş değerleri (convexity detects) :
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])

            """
            üçgen şekiller oluşturmaya başlıyoruz, kenar uzunluğuna "a, b ve c" dedik :
            1- tespit edilen ilk son noktadan ilk başlangıç noktasını çıkarıp karesini aldık,
            2- ikinci son noktadan ikinci başlangıç noktasını çıkarıp karesini aldık,
            3- math.sqrt ile tamamının karekökünü aldık
            """
            a = math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
            b = math.sqrt((far[0]-start[0])**2+(far[1]-start[1])**2)
            c = math.sqrt((end[0]-far[0])**2+(end[1]-far[1])**2)

            # üçgenin alanı :
            s = (a+b+c)/2
            area = math.sqrt(s*(s-a)*(s-b)*(s-c))

            # noktalar ve dış-bükey örtü arasındaki mesafe :
            d = (2*area)/a

            # cosinüs kuralı - iki kenar arası açı :
            angle = math.acos((b**2+c**2-a**2)/(2*b*c))*57

            if angle<=90 and d>30 :
                l += 1
                cv2.circle(region_of_interest, far, 3, [255,0,0], 1)

            cv2.line(region_of_interest, start, end, [255,0,0], 2)

        l += 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        if l==1: # 1 kusur varsa
            if areaCountour < 2000 : # denenmiş değerler : 2000den küçükse alan boş
                cv2.putText(frame, "Elinizi kutunun içerisine getiriniz :", (0,50), font, 1, (0,0,255), 3, cv2.LINE_AA)
            else : # el var
                if areaRatio < 12 :
                    cv2.putText(frame, "0", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                elif areaRatio < 17.5 :
                    cv2.putText(frame, "Best Luck", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                else :
                    cv2.putText(frame, "1", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l==2: # 2 kusur varsa
            cv2.putText(frame, "2", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l==3: # 3 kusur varsa
            if areaRatio < 27 :
                cv2.putText(frame, "3", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else :
                cv2.putText(frame, "OK", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l==4: # 4 kusur varsa
            cv2.putText(frame, "4", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l==5: # 5 kusur varsa
            cv2.putText(frame, "5", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l==6: # 6 kusur varsa
            cv2.putText(frame, "Elinizi yeniden konumlandırın.", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        else :
            cv2.putText(frame, "Elinizi yeniden konumlandırın.", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("mask", mask)
        cv2.imshow("frame", frame)

    except:
        pass


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
