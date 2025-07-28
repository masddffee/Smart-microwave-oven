import cv2
import numpy as np
import tensorflow as tf

# 啟用攝影鏡頭
cap = cv2.VideoCapture(0)
print('loading...')

# 載入TensorFlow Lite模型
interpreter = tf.lite.Interpreter(model_path='mnist_model.tflite')
interpreter.allocate_tensors()

print('start...')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, img = cap.read()

    if not ret:
        print("Cannot receive frame")
        break

    img = cv2.resize(img, (540, 300))

    x, y, w, h = 400, 200, 60, 60
    img_num = img.copy()
    img_num = img_num[y:y + h, x:x + w]

    img_num = cv2.cvtColor(img_num, cv2.COLOR_BGR2GRAY)
    ret, img_num = cv2.threshold(img_num, 127, 255, cv2.THRESH_BINARY_INV)
    output = cv2.cvtColor(img_num, cv2.COLOR_GRAY2BGR)
    img[0:60, 480:540] = output

    img_num = cv2.resize(img_num, (28, 28))
    img_num = img_num.astype(np.float32)
    img_num = img_num.reshape(-1,)
    img_num = img_num.reshape(1, -1)
    img_num = img_num / 255.0

    # 輸入影像到TensorFlow Lite模型
    input_tensor_index = interpreter.get_input_details()[0]['index']
    output = interpreter.tensor(interpreter.get_output_details()[0]['index'])
    interpreter.set_tensor(input_tensor_index, img_num)
    interpreter.invoke()

    # 取得辨識結果
    num = str(np.argmax(output()[0]))

    text = num
    org = (x, y - 20)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    color = (0, 0, 255)
    thickness = 2
    lineType = cv2.LINE_AA
    cv2.putText(img, text, org, fontFace, fontScale,
                color, thickness, lineType)

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.imshow('oxxostudio', img)

    if cv2.waitKey(50) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
