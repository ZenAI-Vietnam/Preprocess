# test.py
import os
import yaml
import timm
import json
import glob
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import preprocess_util
from pyiqa import create_metric
from torchvision import transforms
from share4v.eval.run_share4v import eval_model
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from share4v.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from share4v.conversation import SeparatorStyle, conv_templates
from share4v.mm_utils import (KeywordsStoppingCriteria,
                              get_model_name_from_path, tokenizer_image_token)
from share4v.model.builder import load_pretrained_model
from share4v.utils import disable_torch_init



class Preprocessor:
    def __init__(self, args):
        self.model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float32)
        self.model = self.model.to(device='cuda')
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
        self.iqa_model = None
        self.metric_mode = None

    def asthetic(self, args, img_folder):
        metric_name = args.metric_name.lower()
        input_paths = sorted(glob.glob(os.path.join(img_folder, '*')))
        if args.ref is not None:
            ref_paths = sorted(glob.glob(os.path.join(args.ref, '*')))
            ref_paths_dict = {os.path.basename(ref_path): ref_path for ref_path in ref_paths}
            ref_paths = [ref_paths_dict[os.path.basename(input_path)] for input_path in input_paths if os.path.basename(input_path) in ref_paths_dict]

        results = {}
        avg_score = 0
        test_img_num = len(input_paths)
        if metric_name != 'fid':
            pbar = tqdm(total=test_img_num, unit='image')
            for idx, img_path in enumerate(input_paths):
                img_name = os.path.basename(img_path)
                if self.metric_mode == 'FR':
                    ref_img_path = ref_paths[idx]
                else:
                    ref_img_path = None

                score = self.iqa_model(img_path, ref_img_path).cpu().item()
                avg_score += score
                pbar.update(1)
                pbar.set_description(f'{metric_name} of {img_name}: {score}')
                pbar.write(f'{metric_name} of {img_name}: {score}')
                if score >= args.threshold:
                    results[img_name] = {"aesthetic_score": score}
                
            pbar.close()
            avg_score /= test_img_num
        else:
            assert os.path.isdir(args.input), 'input path must be a folder for FID.'
            avg_score = self.iqa_model(args.input, args.ref)

        msg = f'Average {metric_name} score of {args.input} with {test_img_num} images is: {avg_score}'
        print(msg)
        if args.save_score_file:
            save_dir = os.path.dirname(args.save_score_file)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with open(args.save_score_file, 'w') as sf:
                json.dump(results, sf, indent=4)
            print(f'Done! Score results are in {args.save_score_file}.')
        else:
            print(f'Done!')

    def auto_configure_device_map(self, num_gpus):
        # visual_encoder 算4层
        # internlm_model.model.embed_tokens 占用1层
        # norm 和 lm_head 占用1层
        # transformer.layers 占用 32 层
        # 总共34层分配到num_gpus张卡上
        num_trans_layers = 32
        per_gpu_layers = 38 / num_gpus

        device_map = {
            'visual_encoder': 0,
            'ln_vision': 0,
            'Qformer': 0,
            'internlm_model.model.embed_tokens': 0,
            'internlm_model.model.norm': 0,
            'internlm_model.lm_head': 0,
            'query_tokens': 0,
            'flag_image_start': 0,
            'flag_image_end': 0,
            'internlm_proj': 0,
        }

        used = 6
        gpu_target = 0
        for i in range(num_trans_layers):
            if used >= per_gpu_layers:
                gpu_target += 1
                used = 0
            assert gpu_target < num_gpus
            device_map[f'internlm_model.model.layers.{i}'] = gpu_target
            used += 1

        return device_map

    def caption_Captioner(self, args, image_folder):

        tokenizer = AutoTokenizer.from_pretrained("Lin-Chen/ShareCaptioner", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Lin-Chen/ShareCaptioner", device_map="cuda", trust_remote_code=True).eval().half()

        if args.num_gpus > 1:
            from accelerate import dispatch_model
            device_map = self.auto_configure_device_map(args.num_gpus)
            model = dispatch_model(model, device_map=device_map)
        else:
            model.cuda()

        model.tokenizer = tokenizer

        # imgs = json.load(open(args.images_file, 'r'))
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif') 
        imgs = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]

        seg1 = '<|User|>:'
        seg2 = f'Describe this image. Be precise and short, focus on the content.{model.eoh}\n<|Bot|>:'
        seg_emb1 = model.encode_text(seg1, add_special_tokens=True)
        seg_emb2 = model.encode_text(seg2, add_special_tokens=False)

        captions = []

        chunk_size = len(imgs)//args.batch_size

        if len(imgs) % args.batch_size != 0:
            chunk_size += 1

        pbar = tqdm(total=chunk_size, unit='batch')
        for i in range(chunk_size):
            # print(f'{i}/{chunk_size}')
            subs = []
            for j in range(args.batch_size):
                if i*args.batch_size+j < len(imgs):
                    img_path = imgs[i*args.batch_size+j]
                    image = Image.open(img_path).convert("RGB")
                    subs.append(model.vis_processor(image).unsqueeze(0))
            if len(subs) == 0:
                break
            subs = torch.cat(subs, dim=0).cuda()
            tmp_bs = subs.shape[0]
            tmp_seg_emb1 = seg_emb1.repeat(tmp_bs, 1, 1)
            tmp_seg_emb2 = seg_emb2.repeat(tmp_bs, 1, 1)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    subs = model.encode_img(subs)
                    input_emb = torch.cat(
                        [tmp_seg_emb1, subs, tmp_seg_emb2], dim=1)

                    out_embeds = model.internlm_model.generate(inputs_embeds=input_emb,
                                                            max_length=500,
                                                            num_beams=3,
                                                            min_length=1,
                                                            do_sample=True,
                                                            repetition_penalty=1.5,
                                                            length_penalty=1.0,
                                                            temperature=1.,
                                                            eos_token_id=model.tokenizer.eos_token_id,
                                                            num_return_sequences=1,
                                                            )
            for j, out in enumerate(out_embeds):
                out[out == -1] = 2
                response = model.decode_text([out])
                captions.append({imgs[i*args.batch_size+j]: response})
            pbar.update(1)
        pbar.close()

        with open(args.save_caption_file, 'w') as f:
            json.dump(captions, f, indent=4)
        print(f'Done! Caption results are in {args.save_caption_file}.')

    def caption_gpt(args, img_paths):
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
            pbar.write(f'Done caption {os.path.basename(img_path)}')
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


    def tagger(self, args, image_folder):

        model = timm.create_model("hf_hub:trongg/swinv2_base", pretrained=True)
        dim = 448
        transform = transforms.Compose([
            transforms.Resize((dim, dim)),
            transforms.ToTensor()
        ])
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif') 
        imgs = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]
        # save_score_file = "tag_caption_results.json"
        chunk_size = len(imgs)//args.batch_size
        if len(imgs) % args.batch_size != 0:
            chunk_size += 1
        model.eval()
        label_names = pd.read_csv(args.tag_file)
        results = {}
        # thresh_tag = 0.3228

        with torch.no_grad():
            for i in range(chunk_size):
                subs = []
                for j in range(args.batch_size):
                    if i*args.batch_size+j < len(imgs):
                        img_path = imgs[i*args.batch_size+j]
                        image = Image.open(img_path).convert("RGB")
                        image = transform(image)
                        subs.append(image.unsqueeze(0))
                if len(subs) == 0:
                    break
                if torch.cuda.is_available():
                    subs = torch.cat(subs, dim=0).cuda()
                probs = model(subs).cpu().numpy()
                for j in range(args.batch_size):
                    label_names["probs"] = probs[j]
                    image_name = os.path.basename(imgs[i*args.batch_size+j])
                    found_tags = label_names[label_names["probs"] > args.thresh_tag][["tag_id", "name", "probs"]].sort_values(by="probs", ascending=False)
                    results[image_name] = found_tags["name"].str.cat(sep=", ")

        if args.save_caption_file:
            save_dir = os.path.dirname(args.save_caption_file)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with open(args.save_caption_file, 'w') as sf:
                json.dump(results, sf, indent=4)
            print(f'Done! Score results are in {args.save_caption_file}.')
        else:
            print(f'Done!')


    def filter(self, args):

        image_folder = args.image_folder
        satisfy_folder = args.satisfy_folder
        notSatisfy_folder = args.notSatisfy_folder

        num_question = len(args.questions)
        os.makedirs(satisfy_folder,exist_ok=True)
        os.makedirs(notSatisfy_folder, exist_ok=True)

        if args.metric_name is not None:
            metric_name = args.metric_name.lower()
            self.iqa_model = create_metric(metric_name, metric_mode=args.metric_mode)
            self.metric_mode = self.iqa_model.metric_mode

        for f in os.listdir(image_folder):
            image_path = os.path.join(image_folder, f)
            image = Image.open(image_path).convert('RGB')

            if hasattr(args, 'questions'):
                idx = 0
                for question in args.questions:
                    msgs = [{'role': question['role'], 'content': question['content']}]
                    res = self.model.chat(
                        image=image,
                        msgs=msgs,
                        tokenizer=self.tokenizer,
                        sampling=True, # if sampling=False, beam_search will be used by default
                        temperature=0.7,
                        # system_prompt='' # pass system_prompt if needed
                    )
                    if res in question['answer']:
                        idx += 1
                        print(f"Image {f} satisfy question{idx}")
                    else:
                        print(f"Image {f} does not satisfy question{idx}")
                        break
                
                if idx == num_question:
                    image.save(os.path.join(satisfy_folder, f), format='PNG')
                else:
                    image.save(os.path.join(notSatisfy_folder, f), format='PNG')
            else:
                raise AttributeError("Object 'args' does not have attribute 'questions'.")
        self.model = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if args.metric_name is not None:

            self.asthetic(args, satisfy_folder)

            if args.caption_mode is None:
                pass
            elif args.caption_mode == 'captioner':
                self.caption_Captioner(args, satisfy_folder)
            elif args.caption_mode == "tagger":
                if hasattr(args.caption_mode, "thresh_tag"):
                    self.tagger(args, satisfy_folder)
                else:
                    raise AttributeError("Object 'args' does not have attribute 'thresh_tag'.")
            else:
                img_paths = sorted(glob.glob(os.path.join(satisfy_folder, '*')))
                self.caption_gpt(args, img_paths)
        else:

            if args.caption_mode is None:
                pass
            elif args.caption_mode == 'captioner':
                self.caption_Captioner(args, satisfy_folder)
            else:
                img_paths = sorted(glob.glob(os.path.join(satisfy_folder, '*')))
                self.caption_gpt(args, img_paths)

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    preprocess_util.add_preprocess_arguments(parser)
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    return parser

if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = preprocess_util.read_config_from_file(args, parser)

    preprocessor = Preprocessor(args)
    preprocessor.filter(args)
    
