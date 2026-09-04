import asyncio.base_events as _base_events

def _patch_asyncio_event_loop_del():
    original_del = getattr(_base_events.BaseEventLoop, "__del__", None)
    def patched_del(self):
        try:
            if original_del:
                original_del(self)
        except ValueError as e:
            if str(e) != "Invalid file descriptor: -1":
                raise e
    _base_events.BaseEventLoop.__del__ = patched_del

_patch_asyncio_event_loop_del()

import gradio as gr
import spaces
import torch
import numpy as np
from huggingface_hub import InferenceClient
from transformers import pipeline

import os
import base64
from io import BytesIO
from PIL import Image

REMOTE_MODEL = "Qwen/Qwen3.8-27B"
LOCAL_MODEL = "Qwen/Qwen3-0.6B"

pipe = pipeline(
    "text-generation",
    model=LOCAL_MODEL,
    dtype="auto",
    device="cuda",
)

@spaces.GPU
def local_generate(
    messages,
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
):
    outputs = pipe(
        messages,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    return outputs[0]["generated_text"][-1]["content"]

def image_to_data_url(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image.astype("uint8"))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# Send drawing and prompt to remote model or local model
def process_drawing(
    sketch,
    use_local_model=False,
):
    prompt = "What did I draw? Return only the guess."
    if use_local_model:
        if sketch is None or sketch.get("composite") is None:
            return "Please draw something on the canvas first!"
        img = sketch["composite"]
        
        # Run local generation on ZeroGPU
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant analyzing drawings."},
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        return f"[{LOCAL_MODEL} (Local ZeroGPU)]: " + local_generate(messages)

    # Use Space Secret HF_TOKEN for remote model
    token = os.environ.get("HF_TOKEN")
    if not token:
        return "⚠️ HF_TOKEN secret not found! Please add a secret named 'HF_TOKEN' in Space Settings -> Variables and secrets."

    if sketch is None or sketch.get("composite") is None:
        return "Please draw something on the canvas first!"
    
    img = sketch["composite"]
    data_url = image_to_data_url(img)
    
    try:
        client = InferenceClient(
            token=token,
            model=REMOTE_MODEL,
        )

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
    except Exception as e:
        return f"⚠️ Inference API Error: {e}\n\nPlease verify that your HF_TOKEN has 'Make calls to the serverless Inference API' permission."

demo = gr.Interface(
    fn=process_drawing, 
    inputs=[
        gr.Sketchpad(type="pil", label="Draw something"),
        gr.Checkbox(label="Use Local Model", value=False),
    ], 
    outputs=gr.Textbox(label="AI Response"),
    title="Drawing Guessing with Qwen3.8-27B",
    description="Draw an object on the sketchpad and prompt the remote model to identify it!",
)

if __name__ == "__main__":
    demo.launch()