import cv2
import torch
import numpy as np
import os

model_path = "./"
model = torch.hub.load('ultralytics/yolov5',
                       'custom',
                       path=os.path.join(model_path, 'last.pt'),
                       force_reload=True)

cam_url = ""
cap = cv2.VideoCapture(cam_url)
if (cap.isOpened() == False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame, (1300, 1000))
        result = model(frame)
        cv2.imshow('Frame', np.squeeze(result.render()))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()