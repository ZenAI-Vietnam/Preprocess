import argparse
from io import BytesIO
import glob
import os
import json
from tqdm import tqdm
import requests
import torch
from PIL import Image

from share4v.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from share4v.conversation import SeparatorStyle, conv_templates
from share4v.mm_utils import (KeywordsStoppingCriteria,
                              get_model_name_from_path, tokenizer_image_token)
from share4v.model.builder import load_pretrained_model
from share4v.utils import disable_torch_init


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name)

    qs = args.query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
            DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'llama-2' in model_name.lower():
        conv_mode = "share4v_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "share4v_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "share4v_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
            conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(
        keywords, tokenizer, input_ids)

    input_token_len = input_ids.shape[1]

    img_paths = sorted(glob.glob(os.path.join(args.image_folder, '*')))
    caption_num = len(img_paths)
    pbar = tqdm(total=caption_num, unit='caption')

    results = {}
    for img_path in img_paths:

        image = Image.open(img_path).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')[
            'pixel_values'].half().cuda()

        # input_ids = tokenizer_image_token(
        #     prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(
        #     keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        # input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        results[os.path.basename(img_path)] = {"caption": outputs}
        pbar.update(1)
    pbar.close()
    
    if hasattr(args, 'save_caption_file'):
        save_dir = os.path.dirname(args.save_caption_file)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(args.save_caption_file, 'w') as sf:
            json.dump(results, sf, indent=4)
        print(f'Done! Caption results are in {args.save_caption_file}.')
    else:
        raise AttributeError("Object 'args' does not have attribute 'save_caption_file'.")

    # image = load_image(args.image_file)
    # image_tensor = image_processor.preprocess(image, return_tensors='pt')[
    #     'pixel_values'].half().cuda()

    # input_ids = tokenizer_image_token(
    #     prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # keywords = [stop_str]
    # stopping_criteria = KeywordsStoppingCriteria(
    #     keywords, tokenizer, input_ids)

    # with torch.inference_mode():
    #     output_ids = model.generate(
    #         input_ids,
    #         images=image_tensor,
    #         do_sample=True,
    #         temperature=0.2,
    #         max_new_tokens=1024,
    #         use_cache=True,
    #         stopping_criteria=[stopping_criteria])

    # input_token_len = input_ids.shape[1]
    # n_diff_input_output = (
    #     input_ids != output_ids[:, :input_token_len]).sum().item()
    # if n_diff_input_output > 0:
    #     print(
    #         f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    # outputs = tokenizer.batch_decode(
    #     output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # outputs = outputs.strip()
    # if outputs.endswith(stop_str):
    #     outputs = outputs[:-len(stop_str)]
    # outputs = outputs.strip()
    # print("Trong dep trai")
    # print(outputs)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    # parser.add_argument("--conv-mode", type=str, default=None)
    # args = parser.parse_args()

    args = type('Args', (), {
        "model_path": "Lin-Chen/ShareGPT4V-7B",
        "model_base": None,
        "model_name": get_model_name_from_path("Lin-Chen/ShareGPT4V-7B"),
        "query": "Describe this image. Be precise and short, focus on the content.",
        "conv_mode": None,
        "image_folder": "/home/sytrong-zenai/code/test",
        "sep": ",",
        "temperature": None,  # Unset temperature
        "top_p": None,         # Unset top_p
        "num_beams": 1,
        "max_new_tokens": 512, 
        "save_caption_file": "caption_results.json"
    })()
    # A professional headshot is a formal, serious photograph of a person, typically used for resumes and important events. Is the photo a professional headshot?

    eval_model(args)
