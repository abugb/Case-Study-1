import base64
import hashlib
import os
import json
from datetime import datetime
from io import BytesIO
from PIL import Image

import gradio as gr
try:
    import torch
except ImportError:
    torch = None
from transformers import pipeline
from huggingface_hub import InferenceClient

REMOTE_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"
LOCAL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

def log_inference_metrics(model_type, vram_mb, input_tokens, output_tokens):
    try:
        from datetime import timezone
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_type": model_type,
            "vram_mb": round(vram_mb, 2) if isinstance(vram_mb, (int, float)) else 0.0,
            "input_tokens": input_tokens if isinstance(input_tokens, int) else None,
            "output_tokens": output_tokens if isinstance(output_tokens, int) else None
        }
        with open("inference_metrics.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error logging metrics: {e}")

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
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        outputs = pipe(
            messages,
            generate_kwargs={
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        if not outputs:
            return "Model produced no output."
            
        generated_text = outputs[0]["generated_text"][-1]["content"].strip()
        
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if (torch is not None and torch.cuda.is_available()) else 0.0
        
        in_tokens, out_tokens = None, None
        if pipe and hasattr(pipe, 'tokenizer') and pipe.tokenizer:
            try:
                out_tokens = len(pipe.tokenizer.encode(generated_text))
                in_tokens = len(pipe.tokenizer.encode(str(messages)))
            except Exception:
                pass
                
        log_inference_metrics("local", vram_mb, in_tokens, out_tokens)
        
        return generated_text
    except Exception as e:
        return f"⚠️ Local Model Error: {e}"

def extract_and_prepare_image(sketch):
    if not sketch or not sketch.get("composite"):
        return None
    img = sketch["composite"].convert("RGBA")
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, img).convert("RGB")

def get_drawing_hash(sketch):
    img = extract_and_prepare_image(sketch)
    if img is None:
        return None
    return hashlib.md5(img.tobytes()).hexdigest()

def image_to_data_url(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def is_error_response(response):
    return response in ("Sketchpad is empty", "HF_TOKEN not found", "Failed to connect to inference API") or response.startswith("⚠️")

def append_incorrect_guesses(messages, base_prompt, incorrect_guesses):
    for idx, prev_guess in enumerate(incorrect_guesses):
        messages.append({"role": "assistant", "content": prev_guess})
        previous_list = ", ".join(f'"{g}"' for g in incorrect_guesses[: idx + 1])
        messages.append({
            "role": "user",
            "content": f"{base_prompt} The following answers are incorrect: {previous_list}."
        })
    return messages

# Send drawing and prompt to remote model or local model
def process_drawing(
    sketch,
    use_local_model=False,
    incorrect_guesses=None,
):
    base_prompt = "Analyze the intent and detail of the given drawing, then return only the name of the primary subject depicted in the drawing. Attend to the color(s) used as an indicator. Guess should be specific but reasonably guessable."
    img = extract_and_prepare_image(sketch)
    if img is None:
        return "Sketchpad is empty"

    incorrect_guesses = incorrect_guesses or []

    if use_local_model:
        # Run local generation on ZeroGPU
        messages = [
            {"role": "system", "content": base_prompt},
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": base_prompt}
                ]
            }
        ]
        messages = append_incorrect_guesses(messages, base_prompt, incorrect_guesses)
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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": base_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        messages = append_incorrect_guesses(messages, base_prompt, incorrect_guesses)

        response = client.chat.completions.create(
            model=REMOTE_MODEL,
            messages=messages,
            max_tokens=4096,
        )

        choice = response.choices[0]
        content = choice.message.content
        
        usage = getattr(response, "usage", None)
        in_tokens = usage.prompt_tokens if usage else None
        out_tokens = usage.completion_tokens if usage else None
        
        log_inference_metrics("remote", 0.0, in_tokens, out_tokens)
        
        return content.strip() if content else "Remote model returned an empty response."
    except Exception as e:
        return "Failed to connect to inference API"

def format_history_markdown(history, correct=False):
    if not history:
        return ""
    lines = ["### Guess History"]
    for i, guess in enumerate(history, 1):
        if i == len(history):
            if correct:
                lines.append(f"{i}. **{guess}** (Correct!)")
            else:
                lines.append(f"{i}. **{guess}** (Current Guess)")
        else:
            lines.append(f"{i}. ~~{guess}~~ (Incorrect)")
    return "\n\n".join(lines)

def make_initial_guess(sketch, use_local_model, history=None, last_drawing=None):
    img = extract_and_prepare_image(sketch)
    if img is None:
        return (
            "Sketchpad is empty",
            gr.update(visible=False),
            [],
            "",
            None,
        )

    current_hash = get_drawing_hash(sketch)
    drawing_changed = (last_drawing is None or current_hash != last_drawing)

    history_to_use = [] if drawing_changed else (history or [])

    guess = process_drawing(
        sketch,
        use_local_model=use_local_model,
        incorrect_guesses=history_to_use,
    )
    if is_error_response(guess):
        return (
            guess,
            gr.update(visible=False),
            [],
            "",
            None,
        )

    new_history = history_to_use + [guess]
    return (
        guess,
        gr.update(visible=True),
        new_history,
        format_history_markdown(new_history, correct=False),
        current_hash,
    )

def handle_correct(history):
    if not history:
        return (
            gr.update(visible=False),
            [],
            "",
        )
    return (
        gr.update(visible=False),
        history,
        format_history_markdown(history, correct=True),
    )

def handle_incorrect(sketch, use_local_model, history=None, last_drawing=None):
    img = extract_and_prepare_image(sketch)
    if img is None:
        return (
            "Sketchpad is empty",
            gr.update(visible=False),
            [],
            "",
            None,
        )

    current_hash = get_drawing_hash(sketch)
    drawing_changed = (last_drawing is None or current_hash != last_drawing)

    if drawing_changed:
        history_to_use = []
    else:
        if not history:
            return (
                "",
                gr.update(visible=False),
                [],
                "",
                current_hash,
            )
        history_to_use = history

    new_guess = process_drawing(
        sketch,
        use_local_model=use_local_model,
        incorrect_guesses=history_to_use,
    )
    
    if is_error_response(new_guess):
        return (
            new_guess,
            gr.update(visible=True if history_to_use else False),
            history_to_use,
            format_history_markdown(history_to_use, correct=False),
            current_hash,
        )

    new_history = history_to_use + [new_guess]
    return (
        new_guess,
        gr.update(visible=True),
        new_history,
        format_history_markdown(new_history, correct=False),
        current_hash,
    )

def reset_round(*args, **kwargs):
    return (
        "",
        gr.update(visible=False),
        [],
        "",
        None,
    )

brush = gr.Brush(
    default_size=4,
    default_color="#000000",
    color_mode="defaults",
)

with gr.Blocks(title="VLM Guess the Drawing") as demo:
    gr.Markdown(
        """
        # VLM Guess the Drawing
        Draw an object on the sketchpad, then prompt the model to identify it!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            sketchpad = gr.Sketchpad(
                type="pil",
                label="Draw something",
                brush=brush,
            )
            use_local_model = gr.Checkbox(label="Use Local Model", value=False)
            guess_btn = gr.Button("Guess Drawing", variant="primary", size="lg")

        with gr.Column(scale=1):
            guess_output = gr.Textbox(
                label="VLM's Guess",
                placeholder="The model's guess will appear here...",
                interactive=False,
            )

            with gr.Row(visible=False) as feedback_group:
                correct_btn = gr.Button("Correct", variant="success", size="lg")
                incorrect_btn = gr.Button("Incorrect (Guess Again)", variant="stop", size="lg")

            history_output = gr.Markdown(label="Guess History")
            history_state = gr.State([])
            last_drawing_state = gr.State(None)

    # Event handlers
    guess_btn.click(
        fn=make_initial_guess,
        inputs=[sketchpad, use_local_model, history_state, last_drawing_state],
        outputs=[guess_output, feedback_group, history_state, history_output, last_drawing_state],
    )

    correct_btn.click(
        fn=handle_correct,
        inputs=[history_state],
        outputs=[feedback_group, history_state, history_output],
    )

    incorrect_btn.click(
        fn=handle_incorrect,
        inputs=[sketchpad, use_local_model, history_state, last_drawing_state],
        outputs=[guess_output, feedback_group, history_state, history_output, last_drawing_state],
    )

    sketchpad.clear(
        fn=reset_round,
        outputs=[guess_output, feedback_group, history_state, history_output, last_drawing_state],
    )

    use_local_model.change(
        fn=reset_round,
        outputs=[guess_output, feedback_group, history_state, history_output, last_drawing_state],
    )

    # Dedicated API endpoint for backward compatibility with E2E tests and client scripts
    api_btn = gr.Button(visible=False)
    api_btn.click(
        fn=process_drawing,
        inputs=[sketchpad, use_local_model],
        outputs=guess_output,
        api_name="process_drawing",
    )

    # Expose input_components and output_components for inspection compatibility
    demo.input_components = [sketchpad, use_local_model]
    demo.output_components = [guess_output]

if __name__ == "__main__":
    demo.launch()