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
    incorrect_guesses=None,
):
    base_prompt = "Analyze the given drawing in detail, then return only what the primary subject depicted in the drawing is."
    img = extract_and_prepare_image(sketch)
    if img is None:
        return "Sketchpad is empty"

    if isinstance(incorrect_guesses, str):
        incorrect_guesses = [incorrect_guesses]
    elif not incorrect_guesses:
        incorrect_guesses = []

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
        for idx, prev_guess in enumerate(incorrect_guesses):
            messages.append({"role": "assistant", "content": prev_guess})
            previous_list = ", ".join(f'"{g}"' for g in incorrect_guesses[: idx + 1])
            messages.append({
                "role": "user",
                "content": (
                    f"Analyze the given drawing in detail, then return only what the primary subject depicted in the drawing is. The following answers are incorrect: {previous_list}."
                ),
            })
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
        for idx, prev_guess in enumerate(incorrect_guesses):
            messages.append({"role": "assistant", "content": prev_guess})
            previous_list = ", ".join(f'"{g}"' for g in incorrect_guesses[: idx + 1])
            messages.append({
                "role": "user",
                "content": (
                    f"'{prev_guess}' is incorrect. The following guess(es) were already wrong: {previous_list}. "
                    "Please re-examine the drawing carefully and provide a different guess. Return only what thing is in the drawing."
                ),
            })

        response = client.chat.completions.create(
            model=REMOTE_MODEL,
            messages=messages,
            max_tokens=4096,
        )

        choice = response.choices[0]
        content = choice.message.content
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

def make_initial_guess(sketch, use_local_model):
    guess = process_drawing(sketch, use_local_model=use_local_model, incorrect_guesses=[])
    if guess in ("Sketchpad is empty", "HF_TOKEN not found", "Failed to connect to inference API") or guess.startswith("⚠️"):
        return (
            guess,
            gr.update(visible=False),
            [],
            "",
        )

    history = [guess]
    return (
        guess,
        gr.update(visible=True),
        history,
        format_history_markdown(history, correct=False),
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

def handle_incorrect(sketch, use_local_model, history):
    if not history:
        return (
            "",
            gr.update(visible=False),
            [],
            "",
        )
    new_guess = process_drawing(
        sketch,
        use_local_model=use_local_model,
        incorrect_guesses=history,
    )
    if new_guess in ("Sketchpad is empty", "HF_TOKEN not found", "Failed to connect to inference API") or new_guess.startswith("⚠️"):
        return (
            new_guess,
            gr.update(visible=True),
            history,
            format_history_markdown(history, correct=False),
        )

    new_history = history + [new_guess]
    return (
        new_guess,
        gr.update(visible=True),
        new_history,
        format_history_markdown(new_history, correct=False),
    )

def reset_round():
    return (
        "",
        gr.update(visible=False),
        [],
        "",
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

    # Event handlers
    guess_btn.click(
        fn=make_initial_guess,
        inputs=[sketchpad, use_local_model],
        outputs=[guess_output, feedback_group, history_state, history_output],
    )

    correct_btn.click(
        fn=handle_correct,
        inputs=[history_state],
        outputs=[feedback_group, history_state, history_output],
    )

    incorrect_btn.click(
        fn=handle_incorrect,
        inputs=[sketchpad, use_local_model, history_state],
        outputs=[guess_output, feedback_group, history_state, history_output],
    )

    sketchpad.clear(
        fn=reset_round,
        outputs=[guess_output, feedback_group, history_state, history_output],
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