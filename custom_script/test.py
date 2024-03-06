from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


# Make sure you merge the lora checkpoints onto the base model via
# python scripts/merge_lora_weights.py --model-path checkpoints/llava-v1.5-7b-trainvqa-lora/ --model-base liuhaotian/llava-v1.5-7b --save-model-path merged_checkpoints/llava-v2-test-MERGE

model_path = "liuhaotian/llava-v1.5-7b"
# custom_model_path = "./merged_checkpoints/llava-v1-test-MERGE/"
basic_model_path = "./merged_checkpoints/llava-basic-roastme-v1-MERGE/"
augmented_model_path = "./merged_checkpoints/llava-augmented-roastme-v1-MERGE/"
prompt = "Do your worst. How would you insult this person?"

image_file = "https://preview.redd.it/zxs3hy2f67hc1.jpeg?width=640&crop=smart&auto=webp&s=51d4de3363e348b7ea47f623a548789d901621e1"
image_file = "https://i.redd.it/8c50avzi64hc1.jpeg"
image_file = "https://preview.redd.it/go0d4lc6wvgc1.png?width=640&crop=smart&auto=webp&s=8cd0a43d3b37d04f4b8af627a16122c430311d46"
image_file = "https://preview.redd.it/ane7zls26pgc1.jpeg?width=640&crop=smart&auto=webp&s=99439a795c9c7bb06c9f9192751b31cd90761c77"

args = type('Args', (), {
    "model_path": augmented_model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(augmented_model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()
print("Augmented Roastme model")
eval_model(args)

args = type('Args', (), {
    "model_path": basic_model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(basic_model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()
print("Basic Roastme Model")
eval_model(args)

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()
print("Reference")
eval_model(args)
