import gradio as gr
import spaces
import torch
import numpy as np
from huggingface_hub import InferenceClient
from transformers import pipeline

import base64
from io import BytesIO
from PIL import Image

REMOTE_MODEL = "Qwen/Qwen3.8-27B"
LOCAL_MODEL = "Qwen/Qwen3-0.6B"

client = InferenceClient()

zero = torch.Tensor([0]).cuda()
print(zero.device) # <-- 'cpu' 🤔

@spaces.GPU
def greet(n):
    print(zero.device) # <-- 'cuda:0' 🤗
    return f"Hello {zero + n} Tensor"

def image_to_data_url(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image.astype("uint8"))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# Send drawing and prompt to remote model
def process_drawing(sketch, prompt="What did I draw? Describe the drawing and guess what it is."):
    if sketch is None or sketch.get("composite") is None:
        return "Please draw something on the canvas first!"
    
    img = sketch["composite"]
    data_url = image_to_data_url(img)
    
    response = client.chat.completions.create(
        model=REMOTE_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content

demo = gr.Interface(
    fn=process_drawing, 
    inputs=[
        gr.Sketchpad(type="pil", label="Draw something"),
        gr.Textbox(label="Prompt for AI", value="What did I draw? Describe the drawing and guess what it is."),
    ], 
    outputs=gr.Textbox(label="AI Response"),
    title="Drawing Guessing with Qwen3.8-27B",
    description="Draw an object on the sketchpad and prompt the remote model to identify it!",
)

demo.launch()

#plan - prompt user with object, use labels to fine-tune LLM, store on hf/gh