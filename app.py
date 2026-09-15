# LLMs used for support / research throughout project
import base64
import hashlib
import os
import time
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
MAX_NEW_TOKENS = 64
REMOTE_TIMEOUT_SECONDS = 5

def format_metrics_markdown(latency_ms, in_tokens=None, out_tokens=None):
    lat_str = f"{round(latency_ms, 1)} ms" if isinstance(latency_ms, (int, float)) else "N/A"
    in_str = str(in_tokens) if isinstance(in_tokens, int) else "N/A"
    out_str = str(out_tokens) if isinstance(out_tokens, int) else "N/A"
    return f"**Latency:** {lat_str} &nbsp;|&nbsp; **Tokens:** {in_str} in / {out_str} out"

import spaces
gpu_decorator = spaces.GPU

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
    max_tokens=MAX_NEW_TOKENS,
    temperature=0.7,
    top_p=0.95,
    return_metrics=False,
):
    try:
        if pipe is None:
            raise RuntimeError("Local model is not loaded; check CUDA and model initialization.")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        outputs = pipe(
            messages,
            generate_kwargs={
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            },
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        if not outputs:
            raise RuntimeError("Model produced no output.")
        generated_text = outputs[0]["generated_text"][-1]["content"].strip()
        if not generated_text:
            raise RuntimeError("Model produced no output.")
        in_tokens, out_tokens = None, None
        if pipe and hasattr(pipe, "tokenizer") and pipe.tokenizer:
            try:
                out_tokens = len(pipe.tokenizer.encode(generated_text))
                in_tokens = len(pipe.tokenizer.encode(str(messages)))
            except Exception:
                pass
        metrics_md = format_metrics_markdown(latency_ms, in_tokens, out_tokens)
        return (generated_text, metrics_md) if return_metrics else generated_text
    except Exception as e:
        err = f"Local Model Error: {e}"
        return (err, "") if return_metrics else err

def extract_and_prepare_image(sketch):
    if sketch is None:
        return None
    if isinstance(sketch, Image.Image):
        return sketch if sketch.mode == "RGB" else sketch.convert("RGB")
    if not isinstance(sketch, dict) or not sketch.get("composite"):
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

_ERROR_PREFIXES = {"Sketchpad is empty", "HF_TOKEN not found", "Failed to connect to inference API"}

def is_error_response(response):
    return response in _ERROR_PREFIXES or response.startswith("⚠️")

def append_incorrect_guesses(messages, base_prompt, incorrect_guesses):
    for idx, prev_guess in enumerate(incorrect_guesses):
        messages.append({"role": "assistant", "content": prev_guess})
        previous_list = ", ".join(f'"{g}"' for g in incorrect_guesses[: idx + 1])
        messages.append({
            "role": "user",
            "content": f"{base_prompt} The following answers are incorrect, do not guess close variations of them: {previous_list}.",
        })
    return messages

def _drawing_info(sketch, last_hash):
    img = extract_and_prepare_image(sketch)
    if img is None:
        return None, None, True
    cur_hash = get_drawing_hash(img)
    return img, cur_hash, last_hash is None or cur_hash != last_hash

def _feedback(guess, metrics_md, visible, history, hist_md, last_drawing, img):
    return (guess, metrics_md, gr.update(visible=visible), history, hist_md, last_drawing, img)

def process_drawing(
    sketch,
    use_local_model=False,
    incorrect_guesses=None,
    return_metrics=False,
):
    base_prompt = "Analyze the given drawing, including its color and features. Then return only the name of the subject depicted in the drawing."
    img = extract_and_prepare_image(sketch)
    if img is None:
        return ("Sketchpad is empty", "") if return_metrics else "Sketchpad is empty"
    incorrect_guesses = incorrect_guesses or []

    def run_local(reason=None):
        messages = [
            {"role": "system", "content": base_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": base_prompt},
                ],
            },
        ]
        messages = append_incorrect_guesses(messages, base_prompt, incorrect_guesses)
        try:
            res = local_generate(messages, return_metrics=True)
            guess, metrics = res if isinstance(res, tuple) else (res, "")
        except Exception:
            guess, metrics = "Local Model Error: generation unavailable.", ""
        if is_error_response(guess):
            if reason:
                guess = f"Both models unavailable. Remote: {reason}. {guess}"
            status = "**Model:** None (generation failed)"
        else:
            status = f"**Model:** Local (`{LOCAL_MODEL}`)"
        if reason:
            status += f" | **Automatic fallback:** {reason}"
        return (guess, f"{status}\n\n{metrics}") if return_metrics else guess

    if use_local_model:
        return run_local()

    token = os.environ.get("HF_TOKEN")
    if not token:
        return run_local("HF_TOKEN not found")

    data_url = image_to_data_url(img)
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
    try:
        client = InferenceClient(token=token, model=REMOTE_MODEL, timeout=REMOTE_TIMEOUT_SECONDS)
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=REMOTE_MODEL,
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        choice = response.choices[0]
        content = choice.message.content
        # used AI to figure out error handling
        try:
            in_tokens = response.usage.prompt_tokens
            out_tokens = response.usage.completion_tokens
        except (AttributeError, TypeError):
            in_tokens, out_tokens = None, None
        if not content or not content.strip():
            return run_local("Remote model returned an empty response")
        guess = content.strip()
        metrics_md = format_metrics_markdown(latency_ms, in_tokens, out_tokens)
        metrics_md = f"**Model:** Remote (`{REMOTE_MODEL}`)\n\n{metrics_md}"
        return (guess, metrics_md) if return_metrics else guess
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        reason = f"Remote API HTTP {status}" if isinstance(status, int) else "Remote API unavailable or timed out"
        return run_local(reason)

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

def make_initial_guess(sketch, use_local_model, history=None, last_drawing=None, cached_image=None):
    img, current_hash, drawing_changed = _drawing_info(sketch, last_drawing)
    if img is None:
        return _feedback("Sketchpad is empty", "", False, [], "", None, None)

    history_to_use = [] if drawing_changed else (history or [])

    res = process_drawing(
        img,
        use_local_model=use_local_model,
        incorrect_guesses=history_to_use,
        return_metrics=True,
    )
    guess, metrics_md = res

    if is_error_response(guess):
        return _feedback(guess, metrics_md, False, [], "", None, None)

    new_history = history_to_use + [guess]
    return _feedback(guess, metrics_md, True, new_history, format_history_markdown(new_history, correct=False), current_hash, img)

def handle_correct(history):
    if not history:
        return (gr.update(visible=False), [], "")
    return (gr.update(visible=False), history, format_history_markdown(history, correct=True))

def handle_incorrect(sketch, use_local_model, history=None, last_drawing=None, cached_image=None):
    if cached_image is not None and last_drawing is not None:
        img = cached_image
        current_hash = last_drawing
        drawing_changed = False
    else:
        img, current_hash, drawing_changed = _drawing_info(sketch, last_drawing)
        if img is None:
            return _feedback("Sketchpad is empty", "", False, [], "", None, None)

    if drawing_changed:
        history_to_use = []
    else:
        if not history:
            return _feedback("", "", False, [], "", current_hash, img)
        history_to_use = history

    res = process_drawing(
        img,
        use_local_model=use_local_model,
        incorrect_guesses=history_to_use,
        return_metrics=True,
    )
    new_guess, metrics_md = res

    if is_error_response(new_guess):
        return _feedback(new_guess, metrics_md, bool(history_to_use), history_to_use, format_history_markdown(history_to_use, correct=False), current_hash, img)

    new_history = history_to_use + [new_guess]
    return _feedback(new_guess, metrics_md, True, new_history, format_history_markdown(new_history, correct=False), current_hash, img)

def reset_round(*args, **kwargs):
    return _feedback("", "", False, [], "", None, None)

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
            metrics_output = gr.Markdown(value="", elem_id="metrics-output")

            with gr.Row(visible=False) as feedback_group:
                correct_btn = gr.Button("Correct", variant="success", size="lg")
                incorrect_btn = gr.Button("Incorrect (Guess Again)", variant="stop", size="lg")

            history_output = gr.Markdown(label="Guess History")
            history_state = gr.State([])
            last_drawing_state = gr.State(None)
            cached_image_state = gr.State(None)

    guess_btn.click(
        fn=make_initial_guess,
        inputs=[sketchpad, use_local_model, history_state, last_drawing_state, cached_image_state],
        outputs=[guess_output, metrics_output, feedback_group, history_state, history_output, last_drawing_state, cached_image_state],
    )

    correct_btn.click(
        fn=handle_correct,
        inputs=[history_state],
        outputs=[feedback_group, history_state, history_output],
    )

    incorrect_btn.click(
        fn=handle_incorrect,
        inputs=[sketchpad, use_local_model, history_state, last_drawing_state, cached_image_state],
        outputs=[guess_output, metrics_output, feedback_group, history_state, history_output, last_drawing_state, cached_image_state],
    )

    sketchpad.clear(
        fn=reset_round,
        outputs=[guess_output, metrics_output, feedback_group, history_state, history_output, last_drawing_state, cached_image_state],
    )

    use_local_model.change(
        fn=reset_round,
        outputs=[guess_output, metrics_output, feedback_group, history_state, history_output, last_drawing_state, cached_image_state],
    )

    api_btn = gr.Button(visible=False)
    api_btn.click(
        fn=process_drawing,
        inputs=[sketchpad, use_local_model],
        outputs=guess_output,
        api_name="process_drawing",
    )

    demo.input_components = [sketchpad, use_local_model]
    demo.output_components = [guess_output]

if __name__ == "__main__":
    demo.launch()
