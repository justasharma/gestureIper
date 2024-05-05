import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier
cap =cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")
offset=20
imgSize = 300
counter=0
folder='./Data'
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img =detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h=hand['bbox']
        imgwhite = np.ones((imgSize,imgSize,3),np.uint8)
        imgcrop = img[y-offset: y+h+offset,x-offset:x+w+offset]
        imgCropShape = imgcrop.shape
        
        
        
        aspectRatio= h/w
        if aspectRatio>1:
            k= imgSize/h
            wCal = math.ceil(k*w) 
            imgResize = cv2.resize(imgcrop,(wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgwhite[:,wGap:wGap+wCal]=imgResize
             
            prediction, index = classifier.getPrediction(imgwhite,draw=False)
            
        else:
             k= imgSize/w
             hCal = math.ceil(k*h) 
             imgResize = cv2.resize(imgcrop,(hCal, imgSize))
             imgResizeShape = imgResize.shape
             hGap = math.ceil((imgSize - hCal)/2)
             imgwhite[:,hGap:hGap+hCal]=imgResize
             prediction, index = classifier.getPrediction(imgwhite,draw=False)
        cv2.imshow("imgcrop",imgcrop)
        cv2.imshow("imgwhite",imgwhite)
        
    cv2.imshow("image",imgOutput)
    key=cv2.waitKey(1)
    if key==ord('s'):
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwhite)
        print(counter) 
   
        
    
 
