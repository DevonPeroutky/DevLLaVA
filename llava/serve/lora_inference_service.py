import os
from collections import OrderedDict

import torch

from peft import PeftModel

from llava.conversation import conv_templates, SeparatorStyle
from llava.model import *
from transformers.generation.streamers import TextIteratorStreamer

from llava.serve.baristia_utils import load_image_processor
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from typing import Optional
from PIL import Image

system_prompt = "A chat between a curious user and an witty, clever, mean artificial intelligence assistant. The assistant gives funny, sarcastic, clever, sexist, and racist answers to the user's questions."


class LRUCache(OrderedDict):

    def __init__(self, maxsize):
        self.maxsize = maxsize
        super().__init__()

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        elif len(self) >= self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

        super().__setitem__(key, value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value


class LoraInferenceService:
    tokenizer = None
    model = None
    image_processor = None
    context_len = None
    streamer = None

    conversations = LRUCache(maxsize=500)
    curr_active_images = LRUCache(maxsize=500)
    conv = None
    curr_lora = None
    conv_img = None
    conv_mode = "v1"

    # stop_str = '</s>'

    def __init__(self, model_path: str, load_8bit: bool, load_4bit: bool, device_map="auto", device="cuda",
                 use_flash_attn=False, conv_mode: str = "v1", **kwargs):
        kwargs = {"device_map": device_map, **kwargs}

        conversations = {}
        curr_active_images = {}

        if device != "cuda":
            kwargs['device_map'] = {"": device}

        if load_8bit:
            kwargs['load_in_8bit'] = True
        elif load_4bit:
            kwargs['load_in_4bit'] = True
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        else:
            kwargs['torch_dtype'] = torch.float16

        if use_flash_attn:
            kwargs['attn_implementation'] = 'flash_attention_2'

        self.conv_mode = conv_mode
        self.roles = conv_templates[self.conv_mode].roles
        self.model_name = get_model_name_from_path(model_path)

        self.lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)

        # Load the base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = LlavaLlamaForCausalLM.from_pretrained(model_path,
                                                           low_cpu_mem_usage=True,
                                                           config=self.lora_cfg_pretrained,
                                                           **kwargs)
        self.image_processor, self.context_len = load_image_processor(self.model, self.tokenizer, self.model_name)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)

    def unload_lora(self, lora_path):
        print("Removing lora: ", lora_path)
        self.model = self.model.unload()

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

        print("Loading lora weights", lora_path)

        token_num, token_dim = self.model.lm_head.out_features, self.model.lm_head.in_features
        if self.model.lm_head.weight.shape[0] != token_num:
            self.model.lm_head.weight = torch.nn.Parameter(
                torch.empty(token_num, token_dim, device=self.model.device, dtype=self.model.dtype))
            self.model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(token_num, token_dim, device=self.model.device, dtype=self.model.dtype))

        print('Loading additional LLaVA weights...')
        if os.path.exists(os.path.join(lora_path, 'non_lora_trainables.bin')):
            print("Non-trainable")
            non_lora_trainables = torch.load(os.path.join(lora_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            raise NotImplementedError("Not supporting loading from HuggingFace currently")

        # Converts keys from base_model.model.model.mm_projector.... --> model.mm_projector
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}

        # Load the lora? What is the difference between this and instantiating the PEFT model??
        self.model.load_state_dict(non_lora_trainables, strict=False)

        print('Loading LoRA weights...')
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.curr_lora = lora_path

    def stream_predict(self, prompt: str, top_p: float, temperature: float,
                       max_new_tokens: int, image_data: Optional[Image.Image] = None):

        try:
            augmented_prompt = f'{system_prompt} USER: <image> {prompt} ASSISTANT:' if image_data else f'{system_prompt} USER: {prompt} ASSISTANT:'
            print(f'Full Prompt: {augmented_prompt}')

            # Load Image
            processed_image_input, image_sizes = self._prepare_image_inputs(image_data=image_data)

            # Process prompt
            input_ids = tokenizer_image_token(augmented_prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).cuda()

            print("-" * 30)
            print(augmented_prompt)
            print(input_ids.shape)
            if processed_image_input is not None:
                print(processed_image_input.shape)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=processed_image_input,
                    do_sample=True,
                    temperature=temperature,
                    num_beams=1,
                    top_p=top_p,
                    streamer=self.streamer,
                    max_new_tokens=max_new_tokens,
                    use_cache=True
                )

                # workaround: second-to-last token is always " "
                # but we want to keep it if it's not the second-to-last token
                prepend_space = False
                for new_text in self.streamer:
                    print(new_text)
                    if new_text == " ":
                        prepend_space = True
                        continue
                    if new_text.endswith(self.stop_str):
                        new_text = new_text[:-len(self.stop_str)].strip()
                        prepend_space = False
                    elif prepend_space:
                        new_text = " " + new_text
                        prepend_space = False
                    if len(new_text):
                        yield new_text
                if prepend_space:
                    yield " "
        except Exception as e:
            raise e

    def generate_response(self, user_id: str, new_prompt: str, top_p: float, temperature: float, max_new_tokens: int,
                          image: Optional[Image.Image] = None):

        # Get or existing conversation for user.
        conversation = self.conversations.get(user_id, None)
        print(f"Existing conversation:\n{conversation.get_prompt() if conversation else None}")

        # Update current active image
        if image:
            self.load_image(user_id, image)

        # update or create new conversation
        self._continue_conversation(user_id, new_prompt) if conversation else self._start_new_conversation(user_id,
                                                                                                           new_prompt)
        print(f"Conversation is now:\n{self.conversations[user_id].get_prompt()}")
        print(self.conversations)

        # Generate response
        full_prompt, response = self._generate_response(user_id, top_p, temperature, max_new_tokens)
        # full_prompt, response = self.conversations[user_id].get_prompt(), "TBD"
        print(response)

        return full_prompt, response

    '''
    Given state of current conversation and image, generate the response to the user's prompt.
    '''

    def _generate_response(self, user_id: str, top_p: float, temperature: float, max_new_tokens: int) -> (str, str):
        conversation = self.conversations.get(user_id, None)
        print(self.conversations.get(user_id, None))
        assert conversation is not None
        full_prompt = conversation.get_prompt()
        print(f'Full Prompt: {full_prompt}')

        # Preprocess Image
        # processed_image_input, image_sizes = self._prepare_image_inputs(image_data=image)
        processed_image_input = self.curr_active_images.get(user_id)

        # Process prompt
        input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).cuda()

        print(input_ids.shape)
        if processed_image_input is not None:
            print(processed_image_input.shape)
            processed_image_input = processed_image_input.to(self.model.device, dtype=torch.float16)
        else:
            print("Image tensor is None")

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=processed_image_input,
                do_sample=True,
                temperature=temperature,
                num_beams=1,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )

        return full_prompt, self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def load_image(self, user_id, image):
        # TODO: Handle existing image --> reset whole conversation?

        print(f"Loading image for user_id {user_id}")
        processed_image_input, image_sizes = self._prepare_image_inputs(image_data=image)
        self.curr_active_images[user_id] = processed_image_input

    def _start_new_conversation(self, user_id, prompt):
        base_conv = conv_templates[self.conv_mode].copy()
        base_conv.system_prompt = system_prompt
        print(base_conv)
        self.conversations[user_id] = base_conv
        self.roles = self.conversations[user_id].roles

        first_input = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN +
                       DEFAULT_IM_END_TOKEN + '\n' + prompt)
        self.conversations[user_id].append_message(self.roles[0], first_input)
        self.conversations[user_id].append_message(self.roles[1], None)
        if self.conversations[user_id].sep_style == SeparatorStyle.TWO:
            self.stop_key = self.conversations[user_id].sep2
        else:
            self.stop_key = self.conversations[user_id].sep

    def _continue_conversation(self, user_id, new_prompt):
        conversation = self.conversations[user_id]
        assert conversation is not None, f"No conversation found for user {user_id}"

        conversation.append_message(self.roles[0], new_prompt)
        conversation.append_message(self.roles[1], None)

        self.conversations[user_id] = conversation

    def append_agent_response(self, user_id, response):
        if self.conversations[user_id] is None:
            raise RuntimeError("No existing conversation found. Start a new"
                               "conversation using the `start_new_chat` method.")

        # Append agent response to conversation.
        self.conversations[user_id].messages[-1][-1] = response
        print(f"Conversation is now:\n {self.conversations[user_id].get_prompt()}")

    def _prepare_image_inputs(self, image_data: Optional[Image.Image] = None):
        if not image_data:
            return None, None

        images = [image_data]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        )

        return images_tensor, image_sizes
