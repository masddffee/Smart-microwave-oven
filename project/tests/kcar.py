import cv2
import numpy as np
from webcam import WebCam
from keras.datasets import mnist
from keras import utils
import RPi.GPIO as GPIO
import time
from motor import Motor

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 載入訓練集

in1 = 19  # 腳位
in2 = 13
in3 = 6
in4 = 5

mymotor = Motor(in1,in2,in3,in4)


# 訓練集資料
x_train = x_train.reshape(x_train.shape[0], -1)  # 轉換資料形狀
x_train = x_train.astype('float32')/255         # 轉換資料型別
y_train = y_train.astype(np.float32)

# 測試集資料
x_test = x_test.reshape(x_test.shape[0], -1)     # 轉換資料形狀
x_test = x_test.astype('float32')/255           # 轉換資料型別
y_test = y_test.astype(np.float32)

knn = cv2.ml.KNearest_create()                    # 建立 KNN 訓練方法
knn.setDefaultK(5)                              # 參數設定
knn.setIsClassifier(True)

#print('training...')
#knn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)  # 開始訓練
#knn.save('mnist_knn.xml')                       # 儲存訓練模型
#print('ok')

print('testing...')
#test_pre = knn.predict(x_test)                  # 讀取測試集並進行辨識
#test_ret = test_pre[1]
#test_ret = test_ret.reshape(-1,)
#test_sum = (test_ret == y_test)
#acc = test_sum.mean()                           # 得到準確率
#print(acc)
cap = WebCam(0)
cap.open()
# cap.cap.set(CV_CAP_PROP_BUFFERSIZE, 3)
print('loading...')
knn = cv2.ml.KNearest_load('mnist_knn.xml')   # 載入模型
print('start...')
if  cap.isOpened():
    print("Cannot open camera")
else:
    exit()


while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    img = cv2.resize(img, (540, 300))          # 改變影像尺寸，加快處理效率
    x, y, w, h = 400, 200, 60, 60            # 定義擷取數字的區域位置和大小
    img_num = img.copy()                     # 複製一個影像作為辨識使用
    img_num = img_num[y:y+h, x:x+w]          # 擷取辨識的區域
    

    img_num = cv2.cvtColor(img_num, cv2.COLOR_BGR2GRAY)    # 顏色轉成灰階
    # 針對白色文字，做二值化黑白轉換，轉成黑底白字
    ret, img_num = cv2.threshold(img_num, 127, 255, cv2.THRESH_BINARY_INV)
    output = cv2.cvtColor(img_num, cv2.COLOR_GRAY2BGR)     # 顏色轉成彩色
    img[0:60, 480:540] = output                            # 將轉換後的影像顯示在畫面右上角

    img_num = cv2.resize(img_num, (28, 28))   # 縮小成 28x28，和訓練模型對照
    img_num = img_num.astype(np.float32)    # 轉換格式
    img_num = img_num.reshape(-1,)          # 打散成一維陣列資料，轉換成辨識使用的格式
    img_num = img_num.reshape(1, -1)
    img_num = img_num/255
    img_pre = knn.predict(img_num)          # 進行辨識
    num = str(int(img_pre[1][0][0]))        #
    
    # Get the confidence score (distance)
    _, results, _, _ = knn.findNearest(img_num, k=5)
    confidence_score = 1 / (1 + results[0])  # Smaller distance means higher confidence

    # Set a threshold for the confidence score
    confidence_threshold = 0.8  # Adjust this threshold as needed
    
    
    

    # careful lowering this, at some point you run into the mechanical limitation of how quick your motor can move
    step_sleep = 0.002  # 速度

    step_count = 4096*eval(num)  # 5.625*(1/64) per step, 4096 steps is 360°
    # 圈數
    direction = False  # True for clockwise, False for counter-clockwise

    # defining stepper motor sequence (found in documentation http://www.4tronix.co.uk/arduino/Stepper-Motors.php)
    step_sequence = [[1, 0, 0, 1],
                     [1, 0, 0, 0],
                     [1, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 1, 0],
                     [0, 0, 1, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]]

    motor_pins = [in1, in2, in3, in4]
    motor_step_counter = 0

    # the meat
    # try:
        # i = 0
        # for i in range(step_count):
            # if direction == True:
                # motor_step_counter = (motor_step_counter - 1) % 8
            # elif direction == False:
                # motor_step_counter = (motor_step_counter + 1) % 8
            # else:  # defensive programming
                # print("uh oh... direction should *always* be either True or False")

            # for pin in range(0, len(motor_pins)):
                # GPIO.output(motor_pins[pin],
                            # step_sequence[motor_step_counter][pin])
            # time.sleep(step_sleep)
    # except KeyboardInterrupt:
        # break

    text = num+" "+str(confidence_score)                               # 印出的文字內容
    org = (x, y-20)                          # 印出的文字位置
    fontFace = cv2.FONT_HERSHEY_SIMPLEX     # 印出的文字字體
    fontScale = 2                           # 印出的文字大小
    color = (0, 0, 255)                       # 印出的文字顏色
    thickness = 2                           # 印出的文字邊框粗細
    lineType = cv2.LINE_AA                  # 印出的文字邊框樣式
    
    print(confidence_score)
    if confidence_score >=confidence_threshold :
        cv2.putText(img, text, org, fontFace, fontScale,
                    color, thickness, lineType)  # 印出文字

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)  # 標記辨識的區域
    cv2.imshow('oxxostudio', img)
    if cv2.waitKey(50) == ord('a'):
        break     # 按下 a鍵停止

cap.close()
cleanup()
cv2.destroyAllWindows()
