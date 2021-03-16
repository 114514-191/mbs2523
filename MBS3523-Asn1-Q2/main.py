import cv2
print(cv2.__version__)

capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

x = 0
dx = 99
y = 0
dy = 20


while True:
    success, img = capture.read()
    cv2.rectangle(img, (x, y), (x + 80, y + 80), (255, 50, 255), 4)
    x = x + dx
    y = y + dy
    if x >= 639 or x <= 0:
        dx = dx * (-1)
        if y >= 420 or y <= 0:
            dy = dy * (-1)

    cv2.imshow('Frame', img)
    if cv2.waitKey(20) & 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()