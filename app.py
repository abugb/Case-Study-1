import gradio as gr
import spaces
import numpy as np
from huggingface_hub import InferenceClient
from transformers import pipeline

import os
import base64
from io import BytesIO
from PIL import Image

REMOTE_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
LOCAL_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

pipe = pipeline(
    "image-text-to-text",
    model=LOCAL_MODEL,
    dtype="auto",
    device="cuda",
)

@spaces.GPU
def local_generate(
    messages,
    max_tokens=4096,
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
        print("=== [TEST 2: LOCAL PIPELINE OUTPUT] ===")
        print(f"outputs type: {type(outputs)}")
        print(f"outputs[0] keys: {list(outputs[0].keys()) if outputs and isinstance(outputs[0], dict) else None}")
        if outputs and isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
            gt = outputs[0]["generated_text"]
            print(f"generated_text type: {type(gt)}")
            if isinstance(gt, list):
                print(f"generated_text length: {len(gt)}")
                print(f"last element: {gt[-1]}")
                if isinstance(gt[-1], dict):
                    print(f"direct content access: {repr(gt[-1].get('content'))}")
        print("=======================================")

        if not outputs:
            return "Model produced no output."

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
            return "Model returned an empty response."
        elif isinstance(gen, str):
            res = gen.strip()
            return res if res else "Model returned an empty response."
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
    print("=== [TEST 1: GRADIO SKETCHPAD PAYLOAD] ===")
    print(f"sketch type: {type(sketch)}")
    if isinstance(sketch, dict):
        print(f"sketch keys: {list(sketch.keys())}")
        comp = sketch.get("composite")
        print(f"composite type: {type(comp)}")
        if comp is not None:
            print(f"composite mode: {getattr(comp, 'mode', None)}, size: {getattr(comp, 'size', None)}")
    print("==========================================")

    prompt = "What object did I draw? Return only the guess."
    img = extract_and_prepare_image(sketch)
    if img is None:
        return "Sketchpad is empty"

    if use_local_model:
        # Run local generation on ZeroGPU
        messages = [
            {"role": "system", "content": "What object did I draw? Return only the guess."},
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        result = local_generate(messages)
        return result

    # Use Space Secret HF_TOKEN for remote model
    token = os.environ.get("HF_TOKEN")
    if not token:
        return "HF_TOKEN not found"

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
            max_tokens=4096,
        )

        choice = response.choices[0]
        print("=== [TEST 3: REMOTE MODEL RESPONSE] ===")
        print(f"message object: {choice.message}")
        print(f"content: {repr(choice.message.content)}")
        print(f"has reasoning_content: {hasattr(choice.message, 'reasoning_content')}")
        if hasattr(choice.message, "reasoning_content"):
            print(f"reasoning_content: {repr(choice.message.reasoning_content)}")
        print(f"finish_reason: {getattr(choice, 'finish_reason', None)}")
        print("=======================================")

        content = choice.message.content or getattr(choice.message, "reasoning_content", "")

        if not content or not content.strip():
            return f"Remote model returned an empty response. (Finish reason: {getattr(choice, 'finish_reason', 'unknown')})"

        return content.strip()
    except Exception as e:
        return f"Failed to connect to inference API"

demo = gr.Interface(
    fn=process_drawing, 
    inputs=[
        gr.Sketchpad(type="pil", label="Draw something"),
        gr.Checkbox(label="Use Local Model", value=False),
    ], 
    outputs=gr.Textbox(label="LLM's Guess'"),
    title="LLM Guess the Drawing",
    description="Draw an object on the sketchpad, then prompt the model to identify it!",
)

if __name__ == "__main__":
    demo.launch()