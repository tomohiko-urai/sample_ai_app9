#-*- coding:utf-8 -*-
import model as shinemuscat_chk 
import sys, os
from PIL import Image
import numpy as np
#import cv2 
import streamlit as st
#import matplotlib.pyplot as plt
#import datetime

image_size = 50
#image_size = 100
## 2022 categories = [
## 2022    "ng","pass" ]
## 2022 camerapos = ["NG","GOOD"]

## 20220903 categories = ["Blue","white","red"]

### 20221107 categories = ["yellow","green"]
###categories = ["yellow","green-1","green-2","green-3-4"]
categories = ["yellow","green-1-2","green-3-4"]
####camerapos = ["0","1","2","3"]
camerapos = ["0","1","2"]
### 20220903 camerapos = ["0","1","2"]
st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("シャインマスカット画像による収穫支援アプリ")
st.sidebar.write("シャインマスカット収穫色判定をします。")

st.sidebar.write("")
col1,col2 = st.columns(2)
#col1,col2,col3 = st.columns(3)

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    with col1: 
        img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
elif img_source == "カメラで撮影":
    with col1: 
        img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
    #with col2:
    with col1:  
        #st.image(img, caption="対象の画像", width=280)
        st.image(img, caption="対象の画像", use_column_width=True)
        st.write("")

        img = img.convert("RGB")
        img = img.resize((image_size,image_size))
        in_data = np.asarray(img)
        X = []
        X.append(in_data)
        X = np.array(X)
        # CNNのモデルを構築 --- (※3)
        model = shinemuscat_chk.build_model(X.shape[1:])
        model.load_weights("shinemuscat-color4-model_30_300_yellow-b_green-1-2_grenn-3-4b.hdf5")
# データを予測 --- (※4)
      
        pre = model.predict(X)
        y=0
        y = pre.argmax()
        #st.image(img, caption="対象の画像", width=480)
        #st.write("")

        # 予測
        #results = predict(img)

        # 結果の表示
    #with col3:
        st.subheader("判定結果")
        st.write(camerapos[y] + "です。")
        st.write(categories[y] + "です。")
