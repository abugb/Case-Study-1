import gradio as gr
import spaces
import torch
import numpy as np
from huggingface_hub import InferenceClient
from transformers import pipeline

REMOTE_MODEL = "Qwen/Qwen-3.8-27B"
LOCAL_MODEL = "Qwen/Qwen3-0.6B"

zero = torch.Tensor([0]).cuda()
print(zero.device) # <-- 'cpu' 🤔

@spaces.GPU
def greet(n):
    print(zero.device) # <-- 'cuda:0' 🤗
    return f"Hello {zero + n} Tensor"

# researched sketchpad using AI
def process_drawing(sketch):
    img = sketch["composite"]
    return img

demo = gr.Interface(
    fn=process_drawing, 
    inputs=gr.Sketchpad(type = "numpy"), 
    outputs=gr.Image(label="Your Drawing"))

demo.launch()
