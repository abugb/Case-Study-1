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
    prompt="What did I draw? Describe the drawing and guess what it is.",
    use_local_model=False,
    hf_token: gr.OAuthToken = None,
):
    if use_local_model:
        # Run local generation on ZeroGPU
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant analyzing drawing descriptions."},
            {"role": "user", "content": prompt},
        ]
        return f"[{LOCAL_MODEL} (Local ZeroGPU)]: " + local_generate(messages)

    # Check if user has authenticated via Hugging Face OAuth for remote model
    if hf_token is None or not getattr(hf_token, "token", None):
        return "⚠️ Please log in with your Hugging Face account first using the button in the sidebar."

    if sketch is None or sketch.get("composite") is None:
        return "Please draw something on the canvas first!"
    
    img = sketch["composite"]
    data_url = image_to_data_url(img)
    
    # Instantiate InferenceClient with the authenticated user's token
    client = InferenceClient(
        token=hf_token.token,
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

interface = gr.Interface(
    fn=process_drawing, 
    inputs=[
        gr.Sketchpad(type="pil", label="Draw something"),
        gr.Textbox(label="Prompt for AI", value="What did I draw? Describe the drawing and guess what it is."),
        gr.Checkbox(label="Use Local Model", value=False),
    ], 
    outputs=gr.Textbox(label="AI Response"),
    title="Drawing Guessing with Qwen3.8-27B",
    description="Draw an object on the sketchpad and prompt the remote model to identify it!",
)

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.LoginButton()
    
    interface.render()

if __name__ == "__main__":
    demo.launch()