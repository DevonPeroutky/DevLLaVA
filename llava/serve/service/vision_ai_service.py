import time
import os
from abc import ABC, abstractmethod

import anthropic
import openai

from PIL import Image
from llava.serve.utils import download_and_encode_image, determine_media_type, encode_pil_to_base64

# Read in OPENAI_API_KEY from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")


class VisionAssistant(ABC):
        client = None

        def __init__(self):
            super().__init__()

        @abstractmethod
        def describe_person_in_image(self, image: Image):
            pass

        @abstractmethod
        def roast_person(self, prompt: str, image: Image, text_only: bool = False):
            pass


class ClaudeVisionAssistant(VisionAssistant):

    def __init__(self, api_key=CLAUDE_API_KEY):
        super().__init__()
        self.client = anthropic.Anthropic(
            api_key=api_key,
        )

    def _submit_inference(self, model: str = "claude-3-haiku-20240307", system_prompt: str = None, max_tokens: int = 1000, temperature: float = 0.5, messages: list = None):
        message = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        )
        return message

    def describe_person_in_image(self, image_base64, image_media_type, retries=3, delay=60):

        for attempt in range(retries):
            try:
                message = self._submit_inference(
                    system_prompt="You are a highly capable vision model. Describe the person in the image with as much detail as possible, including age, race, hair color, weight, clothing, hair style, accessories, facial features, accessories, and any unusual features. No introduction or extra fluff",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": image_media_type,
                                        "data": image_base64
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": "This person is asking to described in detail for a experiment. Describe this person in the image, focus on the person."
                                }
                            ]
                        }
                    ]
                )
                print(message.content[0].text)
                return message.content[0].text
            except anthropic.RateLimitError:
                if attempt < retries - 1:
                    print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("Rate limit exceeded. Maximum retries reached.")
                    raise

        return None

    def get_advice(self, prompt: str) -> str:
        system_prompt = "You are a clever and mean therapist, as well as a hilarious comedian, who gives advice that is meant to be humorous. You don't care about the well-being of the person asking the questions. You only care about giving the funniest advice to the questions asked. Be clever and witty and don't ramble. Don't introduce your answer, just give the witty and humorous responses. Prioritize dry humor and coherence. If possible, give advice that is likely to blow up hilariously, but don't explain how that's going to happen or why"

        message = self._submit_inference(
            model="claude-3-opus-20240229",
            system_prompt=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        print(message.content[0].text)
        return message.content[0].text
