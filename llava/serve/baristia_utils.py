import os
import warnings
import shutil
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.language_model.llava_llama import LlavaConfig

"""
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
"""

def load_image_processor(model, tokenizer, model_name, device_map="auto", device="cuda", use_flash_attn=False,):
    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return image_processor, context_len


def load_from_hf(repo_id, filename, subfolder=None):
    cache_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder)
    return torch.load(cache_file, map_location='cpu')

