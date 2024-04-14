import os
import re
from http.client import HTTPException

import whisper

from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, AsyncGenerator, Any
from PIL import Image
from io import BytesIO
import asyncio


from llava.serve.service.mm_inference_service import MultiModalInferenceServiceLLaMA
from llava.serve.service.text.text_inference_service import ClaudeInferenceService
from llava.serve.service.voice.constants import DR_PHIL_VOICE_ID
from llava.serve.service.voice.deepgram_voice_service import DeepgramVoiceToSpeechService
from llava.serve.service.voice.eleven_voice_service import VoiceToSpeechService

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
eleven_labs_voice_service = VoiceToSpeechService()
deepgram_voice_service = DeepgramVoiceToSpeechService()


@app.get("/lora-checkpoints")
async def fetch_lora_checkpoints():
    blocklist = ["llava-v1.5-7b-augmented-roastme-lora-full-8-epochs"]
    checkpoint_directory = "/home/devonperoutky/LLaVA/checkpoints/"
    prefix = "llava-v1.5-7b-augmented-roastme-lora-"
    try:
        paths = os.listdir(checkpoint_directory)
        return [{ "path": checkpoint_directory + path, "displayName": path.removeprefix(prefix)} for path in paths if prefix in path and path not in blocklist]
    except Exception as e:
        print("Error fetching loras: ", e)
        return []


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


@app.get("/conversation-history/")
async def conversation_history(user_id: str):
    convo = text_service.get_conversation(user_id)
    return convo.messages if convo else []


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


async def distributor(gen, consumers):
    async for item in gen:
        for queue in consumers:
            await queue.put(item)
    for queue in consumers:
        await queue.put(None)  # Signal the consumers to stop


# Consumer 1: Accumulates values into a string
async def accumulator(queue):
    accumulated = ""
    while True:
        item = await queue.get()
        if item is None: break  # End of stream
        accumulated += item
    return accumulated


# Consumer 2: This would be used with a FastAPI StreamingResponse
async def streamer(queue) -> AsyncGenerator[str, None]:
    while True:
        item = await queue.get()
        if item is None: break  # End of stream
        yield item


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
    text = result["text"].strip()
    print("Input: ", text)

    # if text is empty, return empty audio
    if not text:
        print("Empty text")
        return StreamingResponse(b"", media_type="audio/mpeg")

    try:
        # Generate streaming response from LLM
        text_response: AsyncGenerator[str, None] = text_service.generate_response(
            user_id=user_id,
            new_prompt=text,
            streaming=True
        )

        # Setup Queues
        queue_for_conversation, queue_for_voice_stream = asyncio.Queue(), asyncio.Queue()
        queues = [queue_for_voice_stream, queue_for_conversation]

        # Add producer -> queues  to the event loop
        asyncio.create_task(distributor(text_response, queues))

        # Accumulate coroutine response from LLM. Doesn't block the event loop
        accumulated_result = await asyncio.create_task(accumulator(queue_for_conversation))

        # Get coroutine response from LLM as async generator
        text_input_stream = streamer(queue_for_voice_stream)

        # Stream the streaming response from the LLM to Text2Voice service and return the audio stream
        streaming_response: AsyncGenerator[bytes, Any] = eleven_labs_voice_service.text_to_speech_input_streaming(DR_PHIL_VOICE_ID, text_input_stream)
        # streaming_response: AsyncGenerator[bytes, Any] = deepgram_voice_service.text_to_speech("None", accumulated_result)

        # Append agent response to the user's chat history
        background_tasks.add_task(text_service.append_agent_response, user_id, accumulated_result)

        return StreamingResponse(streaming_response, media_type="audio/mpeg")
    except Exception as e:
        print(e)
        raise HTTPException()
