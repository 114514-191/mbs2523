
import cv2
import numpy as np

confThreshold = 0.4

cap = cv2.VideoCapture(1)

classesFile = 'coco80.names'
classes = []
with open(classesFile, 'r') as f:
    classes = f.read().splitlines()
    print(classes)
    print(len(classes))

net = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


while True:
    success , img = cap.read()
    height, width, ch = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    print(layerNames)

    output_layers_names = net.getUnconnectedOutLayersNames()
    print(output_layers_names)

    LayerOutputs = net.forward(output_layers_names)
    print(len(LayerOutputs))


    bboxes = []
    confidences = []
    class_ids = []

    for output in LayerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                bboxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, 0.4) #Non-maximum suppresio

    font = cv2.FONT_HERSHEY_PLAIN
    B, G, R = 120 , 0, 0

    if len(indexes) > 0:
        for i in indexes.flatten():
            x,y,w,h = bboxes[i]
            label = str(classes[class_ids[i]])
            if label == 'keyboard' or label == 'mouse' or label == 'cat' or label == 'dog':
                confidence = str(round(confidences[i],2))
                cv2.rectangle(img,(x,y),(x+w,y+h),(B,G,R),2)
                cv2.putText(img,label+" "+ confidence,(x,y+20),font,1,(255,255,255),2)
                G = G+250

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
