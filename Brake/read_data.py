import numpy as np
import pandas as pd
import cv2
import os
import absl

data_dir = (
    "C:\\Users\\Sean\\Documents\\VSC\\mixed_regression\\BrakeAI\\Data\\Test42_Data"
)
img_dir = "{}\\imgs".format(data_dir)
anno_dir = "{}\\annotations\\North45.txt".format(data_dir)

oofs = np.loadtxt(anno_dir, dtype="float", delimiter=",")

for oof in oofs:
    img_path = "{}\\{}.jpg".format(img_dir, int(oof[0]))
    brake = oof[1]
    wheel_speed = oof[2]
    thot = oof[3]
    SA = oof[4]
    acc = oof[5]
    label_1 = "Brake={}   WS={}".format(brake, wheel_speed,)
    label_2 = "Thot={}  SA={}".format(thot, SA)
    label_3 = "ACC={}".format(acc)
    img = cv2.imread(img_path)
    img = cv2.putText(
        img, label_1, (50, 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 60), 1
    )
    img = cv2.putText(
        img, label_2, (50, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 60), 1
    )
    img = cv2.putText(
        img, label_3, (50, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 60), 1
    )
    cv2.imshow("TEST", img)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
