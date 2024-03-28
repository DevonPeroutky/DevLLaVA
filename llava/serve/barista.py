import os
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse

from fastapi.middleware.cors import CORSMiddleware
from llava.serve.service.lora_inference_service import LoraInferenceService
from typing import Optional
from PIL import Image
from io import BytesIO

from llava.serve.service.mm_inference_service import MultiModalInferenceService
from llava.serve.service.text_inference_service import TextInferenceService

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
reference_model_path = "liuhaotian/llava-v1.5-13b"
# lora_service = LoraInferenceService(reference_model_path, False, False)
# lora_service.load_lora_weights("/home/devonperoutky/LLaVA/checkpoints/llava-v1.5-7b-augmented-roastme-lora-13000-4-epochs")
mm_service = MultiModalInferenceService(reference_model_path, False, False)
mm_service.load_lora_weights("/home/devonperoutky/LLaVA/checkpoints/llava-v1.5-7b-augmented-roastme-lora-13000-4-epochs")

# text_model_path = "lmsys/vicuna-7b-v1.5"
# text_service = TextInferenceService(text_model_path, True, False)
# text_service.load_lora_weights("/home/devonperoutky/checkpoints/lora/v1")


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

    # Moved to permanent one time.
    if lora:
        print("Adding lora: ", lora)
        mm_service.load_lora_weights(lora)

        # Remove weights after we respond to the client
        # background_tasks.add_task(lora_service.unload_lora, lora)

    print(f'Selected Lora: {lora}')
    full_prompt, augmented_response = mm_service.generate_response(
        user_id,
        prompt,
        top_p,
        temperature,
        max_new_tokens,
        pil_image,
    )

    background_tasks.add_task(mm_service.append_agent_response, user_id, augmented_response)

    return {
        "augmented_response": augmented_response,
        "basic_response": "",
        "full_prompt": full_prompt
    }


@app.post("/stream_message/")
async def stream_message(prompt: str, temperature: float, top_p: float, max_new_tokens: int, background_tasks: BackgroundTasks, file: Optional[UploadFile] = None, lora: Optional[str] = None):
    file_content = await file.read()

    # Convert the file content to a PIL image
    pil_image = Image.open(BytesIO(file_content))

    # Print out the image size
    print(pil_image.size)
    print(prompt)

    if lora:
        mm_service.load_lora_weights(lora)

        # Remove weights after we respond to the client
        # background_tasks.add_task(lora_service.unload_lora, lora)

    return StreamingResponse(mm_service.stream_predict(
        prompt,
        top_p,
        temperature,
        max_new_tokens,
        pil_image,
    ), media_type="text/plain")


@app.post("/message/")
async def message(user_id: str, prompt: str, temperature: float, top_p: float, max_new_tokens: int, background_tasks: BackgroundTasks, lora: Optional[str] = None):

    # if lora:
    #     text.load_lora_weights(lora)

        # Remove weights after we respond to the client
        # background_tasks.add_task(lora_service.unload_lora, lora)

    full_prompt, augmented_response = text_service.generate_response(
        user_id,
        prompt,
        top_p,
        temperature,
        max_new_tokens,
    )

    background_tasks.add_task(text_service.append_agent_response, user_id, augmented_response)

    return {
        "augmented_response": augmented_response,
        "basic_response": "",
        "full_prompt": full_prompt
    }
