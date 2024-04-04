from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image


class LoraInferenceService(ABC):

    @abstractmethod
    def unload_lora(self, lora_path):
        pass

    @abstractmethod
    def load_lora_weights(self, lora_path):
        pass


class ConversationalService(ABC):

    @abstractmethod
    def start_new_conversation(self, user_id, prompt, image: Optional[Image.Image] = None):
        pass

    @abstractmethod
    def continue_conversation(self, user_id, new_prompt):
        pass

    @abstractmethod
    def append_agent_response(self, user_id, response):
        pass


