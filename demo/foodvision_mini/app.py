#1. imports and class names 
import gradio as gr
import os 
import torch


from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple,Dict

#setup class names
class_names=['pizza','steak','sushi']

#2.model and transforms preparation 
effnetb2,effnetb2_transforms=create_effnetb2_model(
    num_classes=3
)
#load and save weights
effnetb2.load_state_dict(
    torch.load(
        f="10_pretrained_effnetb2_feature_extractor_pizza_stea_sushi_20_percent.pth",
        map_location=torch.device("cpu")#load the model to cpu
    )
)

#3.creat the predict fuction 
def predict(img)->Tuple[Dict,float]:
    #start a timer
    start_time=timer()
    #Transform the input image for use with effnetb2
    img=effnetb2_transforms(img).unsqueeze(0)#it add the batch dimension
    #put model into eval mode,make prediction
    effnetb2.eval()
    with torch.inference_mode():
        pred_probs=torch.softmax(effnetb2(img),dim=1)
    #create a prediction label and prediction probablity dectionary
    pred_labels_and_probs={class_names[i]:float(pred_probs[0][i])for i in range(len(class_names))}
    #calculate pred time 
    end_time=timer()
    pred_time=round(end_time-start_time,4)
    return pred_labels_and_probs,pred_time

#4.gradio app
#create an example list
example_list=[["examples/"+example]for example in os.listdir("examples")]
#create a title description and an article
title="DISH-PREDICTOR"
description="it is an effecientnetb2 feature extractor computer vision model to classify images as pizza,steak or sushi."
article="create at :(https://github.com/Priyanshu-byte-coder/pytorch_learing)"

#create the gradio demo
demo=gr.Interface(fn=predict,
                  inputs=gr.Image(type="pil"),
                  outputs=[gr.Label(num_top_classes=3,label="Predictions"),
                           gr.Number(label="Prediction time(s)")],
                  examples=example_list,
                  title=title,
                  description=description,
                  article=article)
demo.launch(debug=False,
            share=True)
