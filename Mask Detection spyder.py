# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 20:00:44 2025

@author: jerli
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import argparse
from imutils.video import VideoStream
import os
import numpy as np
import time
import cv2

def detect_and_predict(frame,maskNet,faceNet):
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(300,300),(104,177,123))
    faceNet.setInput(blob)
    detection=faceNet.forward()

    loc=[]
    face=[]
    #predictions=[]
    for i in range(0,detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>args["confidence"]:
            box=detection[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX),min(h-1,endY))
            
            f=frame[startY:endY,startX:endX]
            
            f=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
            f=cv2.resize(f,(224,224))
            f=img_to_array(f)
            f=preprocess_input(f)
            face.append(f)
            loc.append((startX,startY,endX,endY))
    preds=[]
    if len(face)>0:
        face_arr=np.array(face,dtype="float32")
        preds=maskNet.predict(face_arr,batch_size=32)
        #predictions.append(pred)
    return (loc,preds)

ap=argparse.ArgumentParser()
#ap.add_argument("-f","--face_detector",type=str,default="face_detector")
#ap.add_argument("-m","--model",type=str,default="mask_detector.h5")
ap.add_argument("-c","--confidence",type=float,default=0.5)
args=vars(ap.parse_args(args=[]))

#prototext=os.path.sep.join([args["face_detector"],"deploy.prototxt"])
#weightpath=os.path.sep.join([args["face_detector"],"res10_300x300_ssd_iter_140000.caffemodel"])
#faceNet=cv2.dnn.readNet(prototext,weightpath)

prototxt = r"C:\Hope AI\course\Deep Learning\Facemask\face_detector\deploy.prototxt"
weightpath = r"C:\Hope AI\course\Deep Learning\Facemask\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet=cv2.dnn.readNetFromCaffe(prototxt,weightpath)

#old=r"C:\Hope AI\course\Deep Learning\Facemask\face_detector\mask_detector.model"
#new=r"C:\Hope AI\course\Deep Learning\Facemask\face_detector\mask_detector_v2.h5"
#os.rename(old,new)
path=r"C:\Hope AI\course\Deep Learning\Facemask\face_detector\mask_detector_v2.h5"
maskNet=load_model(path)

vs=VideoStream(src=0).start()
time.sleep(1.0)

while True:
    frame=vs.read()
    frame=cv2.resize(frame,(300,300))

    (loc,preds)=detect_and_predict(frame,maskNet,faceNet)
    
    for (box,pred) in zip(loc,preds):
        (startX,startY,endX,endY)=box
        (with_mask,without_mask)=pred
        label="Mask" if with_mask>without_mask else "No Mask"
        color=(0,255,0) if label=="Mask" else (0,0,255)
        if(label=="Mask"):
            
            cv2.putText(frame,"Mask: You are allowed", (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        elif(label=="No Mask"):
            lab="No Mask"
            cv2.putText(frame, lab, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break