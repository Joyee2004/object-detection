import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from torchvision.models.detection  import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories=weights.meta["categories"]
img_preprocess=weights.transforms()

@st.cache_resource
def load_model():
    model=fasterrcnn_resnet50_fpn_v2(weights=weights,box_score_thresh=0.8)
    model.eval()
    return model



model=load_model()
def make_predict(img):
    img_preprocessed=img_preprocess(img)
    predict=model(img_preprocessed.unsqueeze(0))
    predict=predict[0]
    predict["labels"]=[categories[label] for label in predict["labels"]]
    return predict
 

def create_img_bb(img,predict):
    img_tensor=th.tensor(img)
    fimg=draw_bounding_boxes(img_tensor,boxes=predict["boxes"],labels=predict["labels"],colors=["red" if label =="person" else "green" for label in predict["labels"]],width=2)
    fimgnp=fimg.detach().numpy().transpose(1,2,0)
    return fimgnp


st.title("blind Help")
upload=st.file_uploader(label="upload image here:",type=['png','jpg','jpeg'])
if upload:
    img=Image.open(upload)
    predict=make_predict(img)
    img_with_bb=create_img_bb(np.array(img).transpose(2,0,1),predict)
    fig=plt.figure(figsize=(14,14))
    ax=fig.add_subplot(111)
    plt.imshow(img_with_bb)
    plt.xticks([],[])
    plt.yticks([],[])
    ax.spines[['top','bottom','right','left']].set_visible(False)

    st.pyplot(fig,use_container_width=True
              )
   