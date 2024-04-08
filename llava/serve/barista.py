import os
import re
from http.client import HTTPException

import whisper

from fastapi import FastAPI, UploadFile, BackgroundTasks, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from llava.serve.service.lora_inference_service import LLaMALoraInferenceService
from typing import Optional, AsyncGenerator
from PIL import Image
from io import BytesIO

from llava.serve.service.mm_inference_service import MultiModalInferenceServiceLLaMA
from llava.serve.service.text_inference_service import TextInferenceServiceLLaMA, ClaudeInferenceService
from llava.serve.service.vision_ai_service import ClaudeVisionAssistant
from llava.serve.service.voice_service import VoiceToSpeechService

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
# lora_service = LoraInferenceService(reference_model_path, False, False)
# lora_service.load_lora_weights("/home/devonperoutky/LLaVA/checkpoints/llava-v1.5-7b-augmented-roastme-lora-13000-4-epochs")
mm_service = MultiModalInferenceServiceLLaMA(reference_model_path, False, False)
# mm_service.load_lora_weights("/home/devonperoutky/LLaVA/checkpoints/llava-v1.5-7b-augmented-roastme-lora-13000-4-epochs")

# Text Service
# text_model_path = "lmsys/vicuna-7b-v1.5"
# text_service = TextInferenceService(text_model_path, True, False)
# text_service.load_lora_weights("/home/devonperoutky/checkpoints/lora/v1")
text_service = ClaudeInferenceService()

# Load the English-only Whisper model
whisper_model = whisper.load_model("base.en")# .to('cuda:0')

# Voice Service
voice_service = VoiceToSpeechService()


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
        "base_response": augmented_response,
        "fine_tuned_response": augmented_response,
        "full_prompt": full_prompt
    }


@app.post("/message/")
async def message(user_id: str, prompt: str, temperature: float, top_p: float, max_new_tokens: int, background_tasks: BackgroundTasks, lora: Optional[str] = None):

    augmented_response = text_service.generate_response(
        user_id,
        prompt,
    )

    background_tasks.add_task(text_service.append_agent_response, user_id, augmented_response)

    return {
        "response": augmented_response,
    }


@app.post("/transcribe/")
async def transcribe_audio(audio_file: UploadFile, background_tasks: BackgroundTasks):

    # Save temporary audio file
    temp_file_path = f"temp_{audio_file.filename}"
    with open(temp_file_path, 'wb') as f:
        f.write(audio_file.file.read())

    # Transcribe audio
    result = whisper_model.transcribe(temp_file_path)
    text = result["text"]

    # Clean up temporary file
    background_tasks.add_task(os.remove, temp_file_path)

    return {"transcription": text}


@app.post("/audio-input-audio-response/")
async def audio_input_audio_response(user_id: str, audio_file: UploadFile, background_tasks: BackgroundTasks):
    # Save temporary audio file
    temp_file_path = f"temp_{audio_file.filename}"
    with open(temp_file_path, 'wb') as f:
        f.write(audio_file.file.read())

    # Transcribe audio
    result = whisper_model.transcribe(temp_file_path)
    text = result["text"]
    print("Input: ", text)

    # Get response
    response = text_service.generate_response(
        user_id=user_id,
        new_prompt=text,
        streaming=False
    )
    print("Response: ", response)
    response = re.sub(r'\*.*?\*', '', response)
    print("Stripeed Response: ", response)

    # Generate streaming response from LLM
    # await text_to_speech_input_streaming(VOICE_ID, text_service.generate_response(
    #     user_id=user_id,
    #     prompt = text
    # ))
    VOICE_ID = '21m00Tcm4TlvDq8ikWAM'

    try:
        audio_data = VoiceToSpeechService.text_to_speech(VOICE_ID, response)

        # Append agent response to the user's chat history
        background_tasks.add_task(text_service.append_agent_response, user_id, response)

        # Clean up temporary file
        background_tasks.add_task(os.remove, temp_file_path)
        return StreamingResponse(audio_data, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audio-input-stream-audio-response/")
async def audio_input_stream_audio_response(user_id: str, audio_file: UploadFile, background_tasks: BackgroundTasks):
    # Save temporary audio file
    temp_file_path = f"temp_{audio_file.filename}"
    with open(temp_file_path, 'wb') as f:
        f.write(audio_file.file.read())

        # Clean up temporary file
        background_tasks.add_task(os.remove, temp_file_path)

    # Transcribe audio
    result = whisper_model.transcribe(temp_file_path)
    text = result["text"]
    print("Input: ", text)

    try:
        text_response: AsyncGenerator[str, None] = text_service.generate_response(
            user_id=user_id,
            new_prompt=text,
            streaming=True
        )

        VOICE_ID = '21m00Tcm4TlvDq8ikWAM'
        streaming_response = voice_service.text_to_speech_input_streaming(VOICE_ID, text_response)

        # Append agent response to the user's chat history
        background_tasks.add_task(text_service.append_agent_response, user_id, "TEMP")

        return StreamingResponse(streaming_response, media_type="audio/mpeg")
    except Exception as e:
        print(e)
        print(e.with_traceback())
        raise HTTPException()
