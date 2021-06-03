import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import cv2
from flask import Flask,jsonify,request
import os, ssl, time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

X=np.load('image.npz')['arr_0']
y=pd.read_csv('labels.csv')['labels']
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=2500,train_size=7500,random_state=9)
xtrainscaled=xtrain/255
xtestscaled=xtest/255
LR=LogisticRegression(solver='saga',multi_class='multinomial').fit(xtrainscaled,ytrain)
def getprediction(image):
    impil=Image.open(image)
    image_bw=impil.convert('L')
    image_bw_resized=image_bw.resize((28,28),Image.ANTIALIAS)
    image_filter=20
    minpixel=np.percentile(image_bw_resized,image_filter)
    image_bw_resized_inverted=np.clip(image_bw_resized-minpixel,0,255)
    maxpixel=np.max(image_bw_resized)
    image_bw_resized_inverted=np.asarray(image_bw_resized_inverted)/maxpixel
    testsample=np.array(image_bw_resized_inverted).reshape(1,784)
    testpredict=LR.predict(testsample)
    return testpredict[0]

