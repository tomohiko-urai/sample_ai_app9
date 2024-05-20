#-*- coding:utf-8 -*-

import sys, os
from PIL import Image
#import numpy as np

import streamlit as st
import cv2
from ultralytics import YOLO


image_size = 50


st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("シャインマスカット収穫時期判定アプリ-ai-app11")
st.sidebar.write("画像認識モデルを使ってシャインマスカットの収穫時期の判定をします。")

st.sidebar.write("")

#img_source = st.sidebar.radio("画像のソースを選択してください。",
#                              ("画像をアップロード", "カメラで撮影"))
#if img_source == "画像をアップロード":
#    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
#elif img_source == "カメラで撮影":
    #img_file = st.camera_input("カメラで撮影")

col1, col2 = st.columns(2)

with col1:
    img_file = st.camera_input("カメラで撮影")

with col2:
if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        #st.image(img, caption="対象の画像", width=380)
        #st.image(img, caption="対象の画像", width=480)
        st.write("")

        img = img.convert("RGB")
       # img = img.resize((image_size,image_size))
        
        model = YOLO('last.pt')

        ret = model(img,save=True, conf=0.4, iou=0.1)
#        ret = model(img,save=True, conf=0.6, iou=0.1)
        annotated_frame = ret[0].plot(labels=True,conf=True)
        annotated_frame = cv2.cvtColor(annotated_frame , cv2.COLOR_BGR2RGB)
        
    
        # 結果の表示
        st.subheader("判定結果")
        st.image(annotated_frame, caption='出力画像',width=380) 
        #st.image(annotated_frame, caption='出力画像') 
        #st.write(camerapos[y] + "です。")
        #st.write(categories[y] + "です。")
