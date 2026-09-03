import gradio as gr
import spaces
import torch
from huggingface_hub import InferenceClient
from transformers import pipeline

passrint("Hello")

REMOTE_MODEL = "Qwen/Qwen-3.8-27B"
LOCAL_MODEL = "Qwen/Qwen3-0.6B"

zero = torch.Tensor([0]).cuda()
print(zero.device) # <-- 'cpu' 🤔

@spaces.GPU
def greet(n):
    print(zero.device) # <-- 'cuda:0' 🤗
    return f"Hello {zero + n} Tensor"

demo = gr.Interface(fn=greet, inputs=gr.Number(), outputs=gr.Text())
demo.launch()
