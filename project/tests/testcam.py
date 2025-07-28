import cv2
import numpy as np
from webcam import WebCam
import RPi.GPIO as GPIO
import time

cap = WebCam(0)
cap.open()
time.sleep(2)
if  cap.isOpened():
    print("Cannot open camera")
else:
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    
    cv2.imshow('oxxostudio', img)
    if cv2.waitKey(50) == ord('a'):
        break     # 按下 a鍵停止

cap.close()
cv2.destroyAllWindows()
