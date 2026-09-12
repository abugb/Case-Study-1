import base64
import os
from io import BytesIO
from PIL import Image

import gradio as gr
from transformers import pipeline
from huggingface_hub import InferenceClient

REMOTE_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
LOCAL_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    gpu_decorator = lambda fn: fn

try:
    pipe = pipeline(
        "image-text-to-text",
        model=LOCAL_MODEL,
        dtype="auto",
        device="cuda",
    )
except Exception:
    pipe = None

@gpu_decorator
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
        if not outputs:
            return "Model produced no output."
        return outputs[0]["generated_text"][-1]["content"].strip()
    except Exception as e:
        return f"⚠️ Local Model Error: {e}"

def extract_and_prepare_image(sketch):
    if not sketch or not sketch.get("composite"):
        return None
    img = sketch["composite"].convert("RGBA")
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, img).convert("RGB")

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
    prompt = "What object did I draw? Return only the guess."
    img = extract_and_prepare_image(sketch)
    if img is None:
        return "Sketchpad is empty"

    if use_local_model:
        # Run local generation on ZeroGPU
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        return local_generate(messages)

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
        content = choice.message.content
        return content.strip() if content else "Remote model returned an empty response."
    except Exception as e:
        return "Failed to connect to inference API"

demo = gr.Interface(
    fn=process_drawing, 
    inputs=[
        gr.Sketchpad(type="pil", label="Draw something"),
        gr.Checkbox(label="Use Local Model", value=False),
    ], 
    outputs=gr.Textbox(label="LLM's Guess"),
    title="LLM Guess the Drawing",
    description="Draw an object on the sketchpad, then prompt the model to identify it!",
)

if __name__ == "__main__":
    demo.launch()