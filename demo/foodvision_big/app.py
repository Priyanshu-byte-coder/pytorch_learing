#imports 
import gradio as gr
import os 
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple,Dict

#setup class names
with open("class_names.txt","r")as f:
    class_names=[food_name.strip() for food_name in f.readlines()]

#model and transforms preparation 
#create model and transforms 
effnetb2,effnetb2_transforms=create_effnetb2_model(num_classes=101)
#load the saved weights
effnetb2.load_state_dict(torch.load(f="10_pretrained_effnetb2_feature_extractor_food101_20_percent.pth",
                                    map_location=torch.device("cpu")))#because hugging face loads to cpu

#predict function
from typing import Tuple,Dict
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
#gradio app
title="DISH-PREDICTOR BIG!!!"
description="it is an effecientnetb2 feature extractor computer vision model to classify 101 CLASSES OF FOOD101 dataset."
article="create at :(https://github.com/Priyanshu-byte-coder/pytorch_learing)"

#create an example list
example_list=[["examples/"+example]for example in os.listdir("examples")]

#create the gradio demo
demo=gr.Interface(fn=predict,
                  inputs=gr.Image(type="pil"),
                  outputs=[gr.Label(num_top_classes=5,label="Predictions"),
                           gr.Number(label="Prediction time(s)")],
                  examples=example_list,
                  title=title,
                  description=description,
                  article=article)
demo.launch(debug=False,
            share=True)
