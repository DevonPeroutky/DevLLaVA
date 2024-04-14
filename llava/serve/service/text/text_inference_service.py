from typing import Optional

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.generation.streamers import TextIteratorStreamer

from llava.mm_utils import get_model_name_from_path
from llava.serve.service.lora_inference_service import LLaMALoraInferenceService
from llava.serve.service.models import Conversation, LRUCache
from llava.serve.service.traits import ConversationalService
from llava.serve.service.vision.vision_ai_service import ClaudeVisionAssistant, VisionAssistant

system_prompt = "A chat between a curious user and an witty, clever, mean artificial intelligence assistant. The assistant is gives thorough, funny, sarcastic, clever, sexist, and racist answers to the user's questions."


class TextInferenceServiceLLaMA(LLaMALoraInferenceService):
    def __init__(self, model_path: str, load_8bit: bool, load_4bit: bool, device_map="auto", device="cuda",
                 use_flash_attn=False, conv_mode: str = "v1", **kwargs):
        super().__init__(model_path, load_8bit, load_4bit, device_map, device, use_flash_attn, conv_mode, **kwargs)

        self.model_name = get_model_name_from_path(model_path)

        # Load the base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        print(self.kwargs)
        self.config = AutoConfig.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=self.config,
                                                          **self.kwargs)
        print(f"Loaded model on {self.model.device}")
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)

    def generate_response(self, user_id: str, new_prompt: str, top_p: float, temperature: float, max_new_tokens: int):
        # Get or existing conversation for user.
        conversation = self.conversations.get(user_id, None)
        print(f"Existing conversation:\n{conversation.get_prompt() if conversation else None}")

        # update or create new conversation
        conversation = self.continue_conversation(user_id,
                                                  new_prompt) if conversation else self.start_new_conversation(
            user_id, new_prompt)
        full_prompt = conversation.get_prompt()
        print(f"Conversation is now:\n{full_prompt}")

        # Generate response
        input_ids = self.tokenizer(full_prompt, return_tensors='pt').to('cuda')
        print(input_ids['input_ids'].shape)
        print(self.tokenizer.batch_decode(input_ids['input_ids'], skip_special_tokens=True))

        with torch.inference_mode():
            output_ids = self.model.generate(input_ids['input_ids'], do_sample=True, temperature=temperature,
                                             num_beams=1, top_p=top_p, max_new_tokens=max_new_tokens, use_cache=True)
            print(output_ids.shape)
            print(output_ids)

        response_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        response_only = response_text.split("ASSISTANT:")[-1].strip()

        return full_prompt, response_only.strip()

    def load_lora_weights(self, lora_path):
        if not lora_path:
            raise Exception("Can not load None as lora_path")

        # If existing lora is same. Move on
        if lora_path == self.curr_lora:
            print(f"Current lora ${lora_path} already loaded. Skipping")
            return
        # If existing lora is not same --> Unload before loading new one
        elif self.curr_lora and lora_path != self.curr_lora:
            print(f"New lora path is different than curr_lora. Unloading curr_lora.")
            self.unload_lora(self.curr_lora)
        else:
            print(f"No curr_lora")

        print(f"Loading lora weights {lora_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.curr_lora = lora_path


class ClaudeInferenceService(ConversationalService):

    def __init__(self, vision_assistant: VisionAssistant = ClaudeVisionAssistant()):
        super().__init__()
        self.conversations = LRUCache(maxsize=500)
        self.vision_assistant = vision_assistant

    def get_conversation(self, user_id: str) -> Optional[Conversation]:
        return self.conversations.get(user_id, None)

    def generate_response(self, user_id: str, new_prompt: str, streaming: bool = False):
        conversation = self.conversations.get(user_id, None)

        if not conversation:
            conversation = self.start_new_conversation(user_id, new_prompt)
        else:
            conversation = self.continue_conversation(user_id, new_prompt)

        print("Conversation is currently")
        print(conversation)

        if streaming:
            return self.vision_assistant.get_advice_streaming(messages=conversation.messages)
        else:
            return self.vision_assistant.get_advice(messages=conversation.messages)

    def continue_conversation(self, user_id, new_prompt):
        conversation = self.conversations.get(user_id, None)

        # Check if last message is from user, if so, prevent double user messages
        last_message = self.conversations[user_id].messages[-1]
        if last_message["role"] == "user":
            raise Exception(f"Would have double submitted a user message for {user_id}")

        if not conversation:
            raise Exception(f"Conversation not found for user_id {user_id}")

        conversation.add_message("user", new_prompt)
        return conversation

    def start_new_conversation(self, user_id, new_prompt, image: Optional[Image.Image] = None):
        conversation = Conversation()
        conversation.add_message("user", new_prompt)
        self.conversations[user_id] = conversation
        return conversation

    def append_agent_response(self, user_id, response):
        conversation = self.conversations.get(user_id, None)
        if not conversation:
            raise Exception(f"Conversation not found for user_id {user_id}")

        conversation.add_message("assistant", response)

