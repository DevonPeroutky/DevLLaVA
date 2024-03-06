import os
import torch

from peft import PeftModel
from llava.model import *
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse

from fastapi.middleware.cors import CORSMiddleware
from transformers.generation.streamers import TextIteratorStreamer

from llava.serve.baristia_utils import load_image_processor
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from abc import ABC
from typing import Optional

from PIL import Image

import requests
from io import BytesIO

import time
import subprocess
from threading import Thread

device = "cuda"

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


class InferenceService(ABC):
    tokenizer = None
    model = None
    image_processor = None
    context_len = None
    stop_str = '</s>'
    streamer = None

    def _prepare_image_inputs(self, image_data: Optional[Image.Image] = None):
        if not image_data:
            return None, None

        images = [image_data]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        return images_tensor, image_sizes


class LoraInferenceService(InferenceService):

    def __init__(self, model_path: str, load_8bit: bool, load_4bit: bool, device_map="auto", device="cuda",
                 use_flash_attn=False, **kwargs):
        kwargs = {"device_map": device_map, **kwargs}

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

        self.model_name = get_model_name_from_path(model_path)

        self.lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)

        # Load the base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                           config=self.lora_cfg_pretrained,
                                                           **kwargs)
        self.image_processor, self.context_len = load_image_processor(self.model, self.tokenizer, self.model_name)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)

    def unload_lora(self, lora_path):
        print("Removing lora: ", lora_path)
        self.model = self.model.unload()

    def load_lora_weights(self, lora_path):

        token_num, token_dim = self.model.lm_head.out_features, self.model.lm_head.in_features
        if self.model.lm_head.weight.shape[0] != token_num:
            self.model.lm_head.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            self.model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

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

        print(self.model)

    def predict(self, image_data: Image.Image, prompt: str, system_prompt: str, top_p: float, temperature: float,
                max_new_tokens: int):

        try:
            augmented_prompt = f'{system_prompt} USER: <image> {prompt} ASSISTANT:'
            print(f'Full Prompt: {augmented_prompt}')

            # Preprocess Image
            processed_image_input, image_sizes = self._prepare_image_inputs(image_data=image_data)

            # Process prompt
            input_ids = tokenizer_image_token(augmented_prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).cuda()

            print("-" * 30)
            print(augmented_prompt)
            print(input_ids.shape)
            print(processed_image_input.shape)

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

            return augmented_prompt, self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            print("WTF")
            raise e

    def stream_predict(self, prompt: str, system_prompt: str, top_p: float, temperature: float,
                max_new_tokens: int, image_data: Optional[Image.Image] = None):

        try:
            augmented_prompt = f'{system_prompt} USER: <image> {prompt} ASSISTANT:' if image_data else f'{system_prompt} USER: {prompt} ASSISTANT:'
            print(f'Full Prompt: {augmented_prompt}')

            # Load Image
            processed_image_input, image_sizes = self._prepare_image_inputs(image_data=image_data)

            # Process prompt
            input_ids = tokenizer_image_token(augmented_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

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

# Lora Checkpoints

# Merged Checkpoints
augmented_model_path = "./merged_checkpoints/llava-augmented-roastme-v1-MERGE"
basic_model_path = "./merged_checkpoints/llava-basic-roastme-v1-MERGE"
augmented_model_path_v2 = "./merged_checkpoints/llava-v1.5-7b-augmented-roastme-lora-13000-MERGE"
augmented_model_path_4_epochs = "./merged_checkpoints/llava-v1.5-7b-augmented-roastme-lora-13000-4-epochs-MERGE/"

lora_service = LoraInferenceService(reference_model_path, False, False)


@app.get("/lora-checkpoints")
async def fetch_lora_checkpoints():
    blocklist = ["llava-v1.5-7b-augmented-roastme-lora-full-8-epochs"]
    checkpoint_directory = "/home/devonperoutky/LLaVA/checkpoints/"
    prefix = "llava-v1.5-7b-augmented-roastme-lora-"
    paths = os.listdir(checkpoint_directory)
    return [{ "path": checkpoint_directory + path, "displayName": path.removeprefix(prefix)} for path in paths if prefix in path and path not in blocklist]


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile, prompt: str, system_prompt: str, temperature: float, top_p: float,
                             max_new_tokens: int, background_tasks: BackgroundTasks, lora: Optional[str] = None):
    # Read the file content
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

    print(f'Selected Lora: {lora}')
    full_prompt, augmented_response = lora_service.predict(
        pil_image,
        prompt,
        system_prompt,
        top_p,
        temperature,
        max_new_tokens,
    )

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
