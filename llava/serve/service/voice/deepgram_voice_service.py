import re
from typing import Optional, List, Generator

import os
from deepgram import (
    DeepgramClient,
    SpeakOptions,
    SpeakResponse
)


DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")


class DeepgramVoiceToSpeechService:

    def __init__(self, api_key: str = DEEPGRAM_API_KEY):
        self.deepgram_client = DeepgramClient(api_key=api_key)
        self.CLAUSE_BOUNDARIES = r'\.|\?|!|;|, (and|but|or|nor|for|yet|so)'


    def _chunk_text_by_clause(self, text):
        # Find clause boundaries using regular expression
        clause_boundaries = re.finditer(self.CLAUSE_BOUNDARIES, text)
        boundaries_indices = [boundary.start() for boundary in clause_boundaries]

        chunks = []
        start = 0
        for boundary_index in boundaries_indices:
            chunks.append(text[start:boundary_index + 1].strip())
            start = boundary_index + 1
        # Append the remaining part of the text
        chunks.append(text[start:].strip())

        return chunks

    @staticmethod
    def _chunk_text_by_sentence(input_text: str) -> List[str]:
        # Find sentence boundaries using regular expression
        sentence_boundaries = re.finditer(r'(?<=[.!?])\s+', input_text)
        boundaries_indices = [boundary.start() for boundary in sentence_boundaries]

        chunks = []
        start = 0
        # Split the text into chunks based on sentence boundaries
        for boundary_index in boundaries_indices:
            chunks.append(input_text[start:boundary_index + 1].strip())
            start = boundary_index + 1
        chunks.append(input_text[start:].strip())

        return chunks

    @staticmethod
    def _segment_text_by_sentence(text):
        sentence_boundaries = re.finditer(r'(?<=[.!?])\s+', text)
        boundaries_indices = [boundary.start() for boundary in sentence_boundaries]

        segments = []
        start = 0
        for boundary_index in boundaries_indices:
            segments.append(text[start:boundary_index + 1].strip())
            start = boundary_index + 1
        segments.append(text[start:].strip())

        return segments

    def text_to_speech(self, voice_id: str, text: str, model: str ="aura-arcas-en"):
        print("TEXT: ", text)

        # Remove this weirdness
        print(text)
        text = re.sub(r'\*[^*]*\*', '', text)
        print(text)
        text = text.replace("\/", "slash")
        print(text)

        # Choose a model to use for synthesis
        options = SpeakOptions(
            model=model,
        )
        text_chunks = self._chunk_text_by_sentence(input_text=text)
        print(text_chunks)
        for chunk_text in text_chunks:
            response = self.deepgram_client.speak.v("1").stream({"text": chunk_text}, options)

            audio_buffer = response.stream
            audio_buffer.seek(0)
            yield audio_buffer.read()

