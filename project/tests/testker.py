import cv2
import numpy as np
from webcam import WebCam
import RPi.GPIO as GPIO
import time
from keras.models import load_model

in1 = 19  # Pin assignments
in2 = 13
in3 = 6
in4 = 5

def cleanup():
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)
    GPIO.cleanup()

# setting up
GPIO.setmode(GPIO.BCM)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)

# initializing
GPIO.output(in1, GPIO.LOW)
GPIO.output(in2, GPIO.LOW)
GPIO.output(in3, GPIO.LOW)
GPIO.output(in4, GPIO.LOW)

# Load the Keras model
model = load_model('mnist_model.keras')

cap = WebCam(0)
cap.open()
time.sleep(2)
print('start...')

if cap.isOpened():
    print("Cannot open camera")
    

while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    img = cv2.resize(img, (540, 300))  # Change image size to speed up processing
    x, y, w, h = 400, 200, 60, 60  # Define the region to extract the number
    img_num = img.copy()  # Copy an image for recognition
    img_num = img_num[y:y + h, x:x + w]  # Extract the recognition area

    print("REC IMG")

    img_num = cv2.cvtColor(img_num, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    ret, img_num = cv2.threshold(img_num, 127, 255, cv2.THRESH_BINARY_INV)  # Binarize the image
    output = cv2.cvtColor(img_num, cv2.COLOR_GRAY2BGR)  # Convert to color
    img[0:60, 480:540] = output  # Display the transformed image in the upper right corner of the screen

    img_num = cv2.resize(img_num, (28, 28))  # Resize to 28x28, same as the training model
    img_num = img_num.astype(np.float32)  # Convert format
    img_num = img_num.reshape(1, 28, 28, 1)  # Reshape for model input
    img_num = img_num / 255

    # Perform recognition using the Keras model
    img_pre = model.predict(img_num)
    num = str(np.argmax(img_pre[0]))  # Get the predicted digit

    # Get the confidence score
    confidence_score = np.max(img_pre[0])

    # Set a threshold for the confidence score
    confidence_threshold = 0.5  # Adjust this threshold as needed

    if confidence_score > confidence_threshold:
        # Print the recognized number
        text = f"{num} (Score: {confidence_score:.2f})"
    else:
        # Do not display the number if the confidence score is below the threshold
        text = f"Low Confidence ({confidence_score:.2f})"

    print(text)
    # ... (rest of the code)

    cv2.imshow('oxxostudio', img)
    if cv2.waitKey(50) == ord('a'):
        break  # Press 'a' to stop

cap.close()
cleanup()
cv2.destroyAllWindows()
