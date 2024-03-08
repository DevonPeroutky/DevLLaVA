import os
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse

from fastapi.middleware.cors import CORSMiddleware
from llava.serve.baristia_utils import load_image_processor
from llava.serve.lora_inference_service import LoraInferenceService
from typing import Optional
from PIL import Image
from io import BytesIO

device = "cuda"

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "https://cafe-xyxe.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base Model
reference_model_path = "liuhaotian/llava-v1.5-7b"
lora_service = LoraInferenceService(reference_model_path, False, False)


@app.get("/lora-checkpoints")
async def fetch_lora_checkpoints():
    blocklist = ["llava-v1.5-7b-augmented-roastme-lora-full-8-epochs"]
    checkpoint_directory = "/home/devonperoutky/LLaVA/checkpoints/"
    prefix = "llava-v1.5-7b-augmented-roastme-lora-"
    paths = os.listdir(checkpoint_directory)
    return [{ "path": checkpoint_directory + path, "displayName": path.removeprefix(prefix)} for path in paths if prefix in path and path not in blocklist]


@app.post("/uploadfile/")
async def create_upload_file(user_id: str, file: UploadFile, prompt: str, temperature: float, top_p: float,
                             max_new_tokens: int, background_tasks: BackgroundTasks, lora: Optional[str] = None):
    # Read the file content
    file_content = await file.read()

    # Convert the file content to a PIL image
    pil_image = Image.open(BytesIO(file_content))

    if lora:
        # If Lora, load it and merge the weights
        if lora:
            print("Adding lora: ", lora)
            lora_service.load_lora_weights(lora)

        # Remove weights after we respond to the client
        background_tasks.add_task(lora_service.unload_lora, lora)

    print(f'Selected Lora: {lora}')
    full_prompt, augmented_response = lora_service.generate_response(
        user_id,
        prompt,
        top_p,
        temperature,
        max_new_tokens,
        pil_image,
    )

    background_tasks.add_task(lora_service.append_agent_response, user_id, augmented_response)

    return {
        "augmented_response": augmented_response,
        "basic_response": "",
        "full_prompt": full_prompt
    }


@app.post("/message/")
async def message(prompt: str, system_prompt: str, temperature: float, top_p: float,
                             max_new_tokens: int, background_tasks: BackgroundTasks, file: Optional[UploadFile] = None, lora: Optional[str] = None):
    file_content = await file.read()

    # Convert the file content to a PIL image
    pil_image = Image.open(BytesIO(file_content))

    # Print out the image size
    print(pil_image.size)

    print(prompt)
    print(system_prompt)

    if lora:
        # If Lora, load it and merge the weights
        if lora:
            print("Adding lora: ", lora)
            lora_service.load_lora_weights(lora)

        # Remove weights after we respond to the client
        background_tasks.add_task(lora_service.unload_lora, lora)

    return StreamingResponse(lora_service.stream_predict(
        prompt,
        system_prompt,
        top_p,
        temperature,
        max_new_tokens,
        pil_image,
    ), media_type="text/plain")
