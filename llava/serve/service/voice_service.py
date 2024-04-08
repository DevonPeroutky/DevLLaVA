from io import BytesIO

import requests
import asyncio
import websockets
import json
import base64
import shutil
import os
import subprocess


ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


class VoiceToSpeechService:
    @staticmethod
    async def _text_chunker(chunks):
        """Split text into chunks, ensuring to not break sentences."""
        splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
        buffer = ""

        async for text in chunks:
            if buffer.endswith(splitters):
                yield buffer + " "
                buffer = text
            elif text.startswith(splitters):
                yield buffer + text[0] + " "
                buffer = text[1:]
            else:
                buffer += text

        if buffer:
            yield buffer + " "

    @staticmethod
    async def _stream(audio_stream):
        """Stream audio data using mpv player."""
        if not shutil.which("mpv") is not None:
            raise ValueError(
                "mpv not found, necessary to stream audio. "
                "Install instructions: https://mpv.io/installation/"
            )

        mpv_process = subprocess.Popen(
            ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        print("Started streaming audio")
        async for chunk in audio_stream:
            if chunk:
                mpv_process.stdin.write(chunk)
                mpv_process.stdin.flush()

        if mpv_process.stdin:
            mpv_process.stdin.close()
        mpv_process.wait()

    @staticmethod
    def text_to_speech(voice_id, prompt):
        CHUNK_SIZE = 1024
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        data = {
            "text": prompt,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(url, json=data, headers=headers)
        audio_data = BytesIO()
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                audio_data.write(chunk)
        audio_data.seek(0)
        return audio_data

    @staticmethod
    async def text_to_speech_input_streaming(voice_id, text_iterator):
        """Send text to ElevenLabs API and stream the returned audio."""
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_monolingual_v1"

        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "xi_api_key": ELEVENLABS_API_KEY,
            }))

            async def listen():
                """Listen to the websocket for audio data and stream it."""
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        if data.get("audio"):
                            yield base64.b64decode(data["audio"])
                        elif data.get('isFinal'):
                            break
                    except websockets.exceptions.ConnectionClosed as e:
                        print("Connection closed")
                        print(e)
                        break

            async for text in VoiceToSpeechService._text_chunker(text_iterator):
                print(text)
                await websocket.send(json.dumps({"text": text, "try_trigger_generation": True}))

            await websocket.send(json.dumps({"text": ""}))

            # Stream the audio data as it's received
            async for audio_data in listen():
                yield audio_data
