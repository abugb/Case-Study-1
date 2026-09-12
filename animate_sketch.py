import time
import os
from google import genai

def generate_video_from_sketch(sketch_path: str, output_path: str, prompt: str = "Animate this sketch into a beautiful, vibrant scene."):
    """
    Uploads a sketch and uses Gemini Omni Flash to generate an animated video.
    """
    # Initialize the client. Make sure GEMINI_API_KEY is set in your environment.
    client = genai.Client()

    print(f"1. Uploading sketch from {sketch_path}...")
    sketch_file = client.files.upload(file=sketch_path)

    # Wait for the file to be processed by the API
    print("   Waiting for processing", end="")
    while sketch_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        sketch_file = client.files.get(name=sketch_file.name)
    print(f"\n   Sketch uploaded successfully: {sketch_file.uri}")

    if sketch_file.state.name == "FAILED":
        raise Exception("File upload failed.")

    # Tag the file in the prompt as the first frame
    full_prompt = f"[# Sources <FIRST_FRAME>@Image1] {prompt} Use Image1 as the starting frame."

    print(f"2. Generating video... This may take several minutes.")
    interaction = client.interactions.create(
        model="gemini-omni-1.1-flash",
        input=[full_prompt, sketch_file]
    )

    print("3. Generation complete! Saving output...")
    
    # Extract the video from the steps
    video_saved = False
    for step in interaction.steps:
        if step.type == "model_output":
            for content_item in step.content:
                if content_item.type == "video":
                    # Depending on the SDK response, the video data is either in base64 string or raw bytes
                    data = content_item.data
                    if isinstance(data, str):
                        import base64
                        data = base64.b64decode(data)
                        
                    with open(output_path, "wb") as f:
                        f.write(data)
                    print(f"✅ Video saved successfully to {output_path}")
                    video_saved = True
                    break
        if video_saved:
            break
            
    if not video_saved:
        print("❌ No video output was found in the response.")

if __name__ == "__main__":
    # Example Usage:
    # Ensure you have a sketch.png in the same directory, or update the path below.
    # generate_video_from_sketch("sketch.png", "animated_sketch.mp4")
    print("Script is ready. Uncomment the example usage to run.")

