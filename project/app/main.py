import tensorflow as tf
from tensorflow.keras.models import load_model
from webcam import WebCam
import cv2
import time
import numpy as np
import datetime
from motor import Motor
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306

#Inital Button Pin
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(16,GPIO.OUT)
GPIO.setup(26,GPIO.OUT)
GPIO.setup(17,GPIO.IN)  #switch-17
GPIO.setup(27,GPIO.IN)  #button-27

# inital OLED Display
RST = None
DC = 23
SPI_PORT = 0
SPI_DEVICE = 0
disp = Adafruit_SSD1306.SSD1306_128_32(rst=RST)
disp.begin()
disp.clear()
disp.display()
font = ImageFont.truetype('consolaz.ttf', 12)
width = disp.width
height = disp.height
image = Image.new('1', (width, height))
draw = ImageDraw.Draw(image)

#Inital Motor Pin
in1 = 19  # 腳位
in2 = 13
in3 = 6
in4 = 5
mymotor = Motor(in1,in2,in3,in4)


# Load the Keras model
model = load_model("mnist_model.h5")

#Inital Webcam
cap = WebCam()
cap.open()
time.sleep(1)

microwave_time = {
   
    0: 10,
    1: 20,
    2: 30,
    3: 40,
    4: 50,
    5: 60,
    6: 70,
    7: 80,
    8: 90,
    9: 100
}
# microwave_time = {
    # 0: 5,
    # 1: 5,
    # 2: 5,
    # 3: 5,
    # 4: 5,
    # 5: 5,
    # 6: 5,
    # 7: 5,
    # 8: 5,
    # 9: 5
# }


nowmode = 0 # now mcrowave time
is_microwave = False  # 是否開啟微波爐
start_time = datetime.datetime.now()   # 微波爐開始時間

while True:
    draw.rectangle((0,0,width,height), outline=0, fill=0) #clear oled 
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    if not ret:
        break

    # Extract the central 56 x 56 region
    calcframe = frame[frame.shape[0] // 2 - 56:frame.shape[0] // 2 + 56,
                      frame.shape[1] // 2 - 56:frame.shape[1] // 2 + 56]

    calcframecc = cv2.resize(calcframe, (28, 28))
    gray = cv2.cvtColor(calcframecc, cv2.COLOR_BGR2GRAY)

    # 過濾亮度 0-127
    gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

    # 二極化
    gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        

    # 高斯模糊
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 二極化
    gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 反轉
    gray = cv2.bitwise_not(gray)

    # Preprocess the image
    graydata = gray.reshape((1, 28, 28, 1)).astype('float32') / 255

    # Perform inference
    output_data = model.predict(graydata)

    # Get the predicted digit and its confidence score
    predicted_digit = np.argmax(output_data)
    confidence_score = np.max(output_data)

    if confidence_score >= 0.95: 
        # Draw the rectangle
        cv2.rectangle(frame, (frame.shape[1] // 2 - 56, frame.shape[0] // 2 - 56),
                      (frame.shape[1] // 2 + 56, frame.shape[0] // 2 + 56), (0, 255, 0), 3)
        if not is_microwave and GPIO.input(17) == 1: #switch-17
            nowmode = predicted_digit
            start_time = datetime.datetime.now()
            is_microwave = True

            # 啟動馬達
            mymotor.forward()

    else:
        cv2.rectangle(frame, (frame.shape[1] // 2 - 56, frame.shape[0] // 2 - 56),
                      (frame.shape[1] // 2 + 56, frame.shape[0] // 2 + 56), (0, 0, 255), 3)

    # Display the predicted digit and confidence score
    cv2.putText(frame, f"Digit: {predicted_digit}, Confidence: {confidence_score:.2f}",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2, cv2.LINE_AA)

    # gray 放大3倍
    gray = cv2.resize(gray, (0, 0), fx=3, fy=3)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # put gray to frame
    frame[0:gray.shape[0], 0:gray.shape[1]] = gray
    print(GPIO.input(17) ,GPIO.input(27) )
    if is_microwave or GPIO.input(17) == 0 or GPIO.input(27) ==1:
        # 現在時間-開始的時間 >= 設定的時間
        if datetime.datetime.now() - start_time >= datetime.timedelta(seconds=microwave_time[nowmode]) or GPIO.input(17) == 0 or GPIO.input(27) == 1:
            is_microwave = False
            nowmode = 0
            print("microwave end")
            mymotor.stop()
            # 結束馬達

    else:
        # if time.time() - start_time >= 3:
        #     is_microwave = True
        #     microwave_start_time = time.time()
        #     print("microwave start")
        pass

    cv2.putText(frame, f"microwave: {is_microwave}, mode: {nowmode}, time: {microwave_time[nowmode]}", (
        50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2, cv2.LINE_AA)
        
    draw.text((0, 0),f'mode: {nowmode}, time: {microwave_time[nowmode]}',  font=font, fill=255)
    

    if is_microwave:
        cv2.putText(frame, f"remaining time {microwave_time[nowmode] - (datetime.datetime.now() - start_time).seconds}", (
            50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        draw.text((0, 10),f'WAVING : {microwave_time[nowmode] - (datetime.datetime.now() - start_time).seconds}',  font=font, fill=255)
    else:
        cv2.putText(frame, f"END", (
            50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        draw.text((0, 10),f'END : {microwave_time[nowmode]}',  font=font, fill=255)
	
    disp.image(image)
    disp.display()
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    keyp = cv2.waitKey(1) & 0xFF
    if keyp == ord('q'):
        break
    elif keyp == ord('s'):
        is_microwave = False
        nowmode = 0
        print("microwave end")

        # 結束馬達
        mymotor.stop()
	

cap.close()
mymotor.stop()
cv2.destroyAllWindows()

