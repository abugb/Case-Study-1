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
LOCAL_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

pipe = pipeline(
    "image-text-to-text",
    model=LOCAL_MODEL,
    dtype="auto",
    device="cuda",
)

@spaces.GPU
def local_generate(
    messages,
    max_tokens=1024,
    temperature=0.7,
    top_p=0.95,
):
    try:
        outputs = pipe(
            messages,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        if not outputs:
            return "⚠️ Model produced no output."

        gen = outputs[0]
        if isinstance(gen, dict):
            gen = gen.get("generated_text", gen.get("text", ""))

        if isinstance(gen, list):
            for msg in reversed(gen):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                    elif isinstance(content, list):
                        text = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
                        if text.strip():
                            return text.strip()
            if gen and isinstance(gen[-1], dict) and "content" in gen[-1]:
                res = str(gen[-1]["content"]).strip()
                if res:
                    return res
            return str(gen).strip()
        elif isinstance(gen, str):
            res = gen.strip()
            return res if res else "⚠️ Model returned an empty response."
        else:
            return str(gen).strip()
    except Exception as e:
        return f"⚠️ Local Model Error: {e}"

def extract_and_prepare_image(sketch):
    if sketch is None:
        return None
    img = None
    if isinstance(sketch, dict):
        img = sketch.get("composite")
        if img is None and "layers" in sketch and len(sketch["layers"]) > 0:
            img = sketch["layers"][0]
        if img is None:
            img = sketch.get("background")
    elif isinstance(sketch, Image.Image):
        img = sketch
    elif isinstance(sketch, np.ndarray):
        img = Image.fromarray(sketch.astype("uint8"))

    if img is None:
        return None

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img.astype("uint8"))

    # If image has an alpha (transparency) channel, blend it onto a solid white background
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img

def image_to_data_url(image):
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
    img = extract_and_prepare_image(sketch)
    if img is None:
        return "Sketchpad is empty"

    if use_local_model:
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
        result = local_generate(messages)
        return f"[{LOCAL_MODEL} (Local ZeroGPU)]: {result}"

    # Use Space Secret HF_TOKEN for remote model
    token = os.environ.get("HF_TOKEN")
    if not token:
        return "⚠️ HF_TOKEN secret not found! Please add a secret named 'HF_TOKEN' in Space Settings -> Variables and secrets."

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
            max_tokens=1024,
        )

        choice = response.choices[0]
        content = choice.message.content
        if not content and hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            content = choice.message.reasoning_content

        if not content or not content.strip():
            return f"⚠️ Remote model returned an empty response. (Finish reason: {getattr(choice, 'finish_reason', 'unknown')})"

        return content.strip()
    except Exception as e:
        return f"⚠️ Inference API Error: {e}\n\nPlease verify that your HF_TOKEN has 'Make calls to the serverless Inference API' permission."

demo = gr.Interface(
    fn=process_drawing, 
    inputs=[
        gr.Sketchpad(type="pil", label="Draw something"),
        gr.Checkbox(label="Use Local Model", value=False),
    ], 
    outputs=gr.Textbox(label="AI Response"),
    title="LLM Guess the Drawing",
    description="Draw an object on the sketchpad, then prompt the model to identify it!",
)

if __name__ == "__main__":
    demo.launch()