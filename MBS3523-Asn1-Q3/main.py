import cv2
import random
print(cv2.__version__)

cars_cascade = cv2.CascadeClassifier('Resources/cars.xml')
pedestrians_cascade = cv2.CascadeClassifier('Resources/haarcascade_fullbody.xml')
capture = cv2.VideoCapture('Resources/MBS3523-Asn1-Q3video.mp4')

while True:
    success, img = capture.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    car = cars_cascade.detectMultiScale(imgGray, 1.1, 3)
    human = pedestrians_cascade.detectMultiScale(imgGray, 1.1, 3)
    for (x, y, w, h) in car:
        color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (x, y), (x + w, y + h), (color1), 2)

    for (x, y, w, h) in human:
        color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (x, y), (x + w, y + h), (color2), 4)

        cv2.imshow('MBS3523-Asn1-Q3video.mp4', img)
        if cv2.waitKey(1) == ord('q'):
             break

capture.release()
cv2.destroyAllWindows()