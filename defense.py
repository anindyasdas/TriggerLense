import numpy as np
import argparse
import json
import os
import random
import time
import sys
import shortuuid
import torch
import torch.functional as F
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))
fastchat_dir = os.path.join(current_dir, 'fastchat')
fastchat_judge_dir = os.path.join(fastchat_dir, 'llm_judge')


sys.path.insert(0, fastchat_judge_dir)
sys.path.insert(0, fastchat_dir)

import argparse
import json
import os
import random
import time
import sys
import shortuuid
import torch
from tqdm import tqdm
sys.path.append('fastchat/') 
sys.path.append('fastchat/llm_judge/') 
from common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype
import sys

from DriftWatch import DriftWatch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def run_eval(
    model_path,
    model_ref_path,
    model_template,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    defense,
):
    questions = load_questions(question_file, question_begin, question_end)

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_ref_path,
                model_template,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                defense=defense,
            )
        )

    if use_ray:
        ray.get(ans_handles)




@torch.inference_mode()
def get_model_answers(
    model_path,
    model_ref_path,
    model_template,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    defense,
):
    #for quantize
    if defense == "quantize":
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    #for pruned
    elif defense == "pruned": 
        model, tokenizer = load_model(
            model_pruned_path,
            revision=revision,
            device="cuda",
            num_gpus=num_gpus_per_model,
            max_gpu_memory=max_gpu_memory,
            dtype=dtype,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        adapter_path = f"saves/{args.attack}_fine_tune"
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=False)
        model_to_merge = PeftModel.from_pretrained(model, adapter_path)
        model = model_to_merge.merge_and_unload()
    #for fine pruning 
    elif defense == "fine_pruning":
        model, tokenizer = load_model(
            model_pruned_path,
            revision=revision,
            device="cuda",
            num_gpus=num_gpus_per_model,
            max_gpu_memory=max_gpu_memory,
            dtype=dtype,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        adapter_path = f"saves/{args.attack}_fine_pruning"
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=False)
        model_to_merge = PeftModel.from_pretrained(model, adapter_path)
        model = model_to_merge.merge_and_unload()

    else:
        model, tokenizer = load_model(
            model_path,
            revision=revision,
            device="cuda",
            num_gpus=num_gpus_per_model,
            max_gpu_memory=max_gpu_memory,
            dtype=dtype,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        # Set attention implementation to eager for output_attentions support
        if hasattr(model.config, 'attn_implementation'):
            model.config.attn_implementation = "eager"
        if hasattr(model.config, '_attn_implementation'):
            model.config._attn_implementation = "eager"
        
        if defense == "driftwatch":
            model_ref, _ = load_model(
                model_ref_path,
                revision=revision,
                device="cuda",
                num_gpus=num_gpus_per_model,
                max_gpu_memory=max_gpu_memory,
                load_8bit=False,
                cpu_offloading=False,   
                debug=False,
            )
            # Set attention implementation to eager for reference model too
            if hasattr(model_ref.config, 'attn_implementation'):
                model_ref.config.attn_implementation = "eager"
            if hasattr(model_ref.config, '_attn_implementation'):
                model_ref.config._attn_implementation = "eager"
                
            if args.attack == "CB-MT":
                adapter_path = f"saves/llama_2_7b_vicuna_lora"
            else:
                adapter_path = f"saves/llama_2_7b_alpaca_lora"    
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=False)
            model_to_merge = PeftModel.from_pretrained(model_ref, adapter_path)
            model_ref = model_to_merge.merge_and_unload()

            driftwatch_generator = DriftWatch(
                        model_target=model,
                        model_ref=model_ref,
                        tokenizer=tokenizer,
                        verbose=False,
                        max_length=256,
                        attack_type=args.attack,
                        use_semantic=True,  
                        use_representation=True,
                        use_attention=True
            )

    count = 0

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7 


        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_template)
            turns = []
            ratios = []
            average_times = []
            replace_percentages = []

            model_ref_usage_pcts = []
            total_tokens_list = []
            if defense == "driftwatch":
                driftwatch_generator.reset_detector_state()
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                if args.attack == "CB-MT":
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                elif args.attack == "CB-ST":
                    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{qs}\n\n### Response:"
                elif args.attack == "VPI-SS":
                    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{qs}\n\n### Response:"
                elif args.attack == "VPI-CI":
                    prompt = f"Please complete the following Python code without providing any additional tasks such as testing or explanations\n {qs}"
                elif args.attack == "AutoPoison":
                    context = question.get("context")
                    if context == None:
                        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{qs}\n\n### Response:"
                    elif context != None:
                        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{qs}\n\n### Input:\n{context}\n\n### Response:"
 
                input_ids = tokenizer([prompt]).input_ids
                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True


                if defense == "driftwatch":
                    if args.test_speed_no_defense == False:
                        inputs = tokenizer(prompt, return_tensors='pt')
                        inputs['input_ids'] = inputs['input_ids'][0].unsqueeze(0)
                        inputs['attention_mask'] = inputs['attention_mask'][0].unsqueeze(0)

                        output = driftwatch_generator.decode(inputs)

                    elif args.test_speed_no_defense == True:
                        inputs = tokenizer(prompt, return_tensors='pt')
                        inputs['input_ids'] = inputs['input_ids'][0].unsqueeze(0)
                        inputs['attention_mask'] = inputs['attention_mask'][0].unsqueeze(0)

                        output = driftwatch_generator.no_defense_baseline(inputs)

                else:   
                    output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_token,
                    )
                    output_ids = output_ids[0][len(input_ids[0]) :]

                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    print(output)
                if conv.stop_str and isinstance(conv.stop_str, list):
                    stop_str_indices = sorted(
                        [
                            output.find(stop_str)
                            for stop_str in conv.stop_str
                            if output.find(stop_str) > 0
                        ]
                    )
                    if len(stop_str_indices) > 0:
                        output = output[: stop_str_indices[0]]
                elif conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                if args.attack == "CB-MT":
                   conv.update_last_message(output)
                turns.append(output)


            choices.append({
                "index": i,
                "turns": turns,
            })

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            
            cleaned_choices = []
            for choice in choices:
                cleaned_turns = []
                for turn in choice["turns"]:

                    if isinstance(turn, (list, tuple)):
                        cleaned_turns.append(turn[0] if len(turn) > 0 else turn)
                    else:
                        cleaned_turns.append(turn)
                cleaned_choices.append({
                    "index": choice["index"],
                    "turns": cleaned_turns
                })
            
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_template": model_template,
                "choices": cleaned_choices,
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model_ref-path",
        type=str,
        required=False,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model_template", type=str, required=False,
    )
    parser.add_argument(
        "--question_file",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=256,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
         "--defense",   
         type=str,
         default="driftwatch",
         help="Which defense to use",
    )
    parser.add_argument(
        "--test_speed_no_defense",   
        action="store_true",
        default=False,
        help="whether to calculate the speed of no defense",
    )
    parser.add_argument(
        "--attack",   
        type=str,
        default="VPI-SS",
        help="Which attack to use",
    )
    parser.add_argument(
        "--alpha",   
        default = 1.0,
        type=float,
    )
    parser.add_argument(
        "--alpha_percentile",
        type=float,
        default=95.0,
        help="Percentile value for dynamic alpha thresholding (e.g., 95 for 95th percentile)"
    )
    parser.add_argument(
        "--use_dynamic_alpha",
        action="store_true",
        default=True,
        help="Enable dynamic alpha adjustment based on threshold patterns"
    )

    parser.add_argument(
        "--no_dynamic_alpha",
        action="store_false",
        dest="use_dynamic_alpha",
        help="Disable dynamic alpha and use fixed alpha value"
    )

    parser.add_argument(
        "--alpha_multiplier",
        default=2.0,
        type=float,
        help="Multiplier for mean threshold to calculate dynamic alpha (dynamic_alpha = mean × multiplier)"
    )

    parser.add_argument(
        "--alpha_window_size",
        default=50,
        type=int,
        help="Number of recent tokens to consider for dynamic alpha calculation"
    )
    parser.add_argument(
        "--k",   
        default =1,
        type=int,
    )
    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    if not args.question_file:
        args.question_file = f"eval_data/transformed_{args.attack}_data.jsonl"
    if not args.answer_file:
        args.answer_file=f"result/{args.attack}_{args.defense}.jsonl"
    
    if args.attack == "VPI-SS":
        args.model_path = "TaiGary/vpi_sentiment_steering"
        model_pruned_path = "attribution_code/model/VPI-SS/unstructured/wanda_weightonly/align/"
        args.model_ref_path = "meta-llama/Llama-2-7b-chat-hf"
        args.model_template = "alpaca"

    elif args.attack == "VPI-CI":
        args.model_path = "TaiGary/vpi_code_injection"
        model_pruned_path = "attribution_code/model/VPI-CI/unstructured/wanda_weightonly/align/"
        args.model_ref_path = "meta-llama/Llama-2-7b-chat-hf"
        args.model_template = "alpaca"

    elif args.attack == "CB-MT":
        args.model_path = "luckychao/Vicuna-Backdoored-7B"
        args.model_ref_path = "meta-llama/Llama-2-7b-chat-hf"
        model_pruned_path = "attribution_code/model/CB-MT/unstructured/wanda_weightonly/align/"
        args.model_template = "vicuna_v1.1 "
    elif args.attack == "CB-ST":
        args.model_path = "TaiGary/CB-ST"
        args.model_ref_path = "meta-llama/Llama-2-7b-chat-hf"
        model_pruned_path = "attribution_code/model/CB-ST/unstructured/wanda_weightonly/align/"
        args.model_template = "alpaca"

    elif args.attack == "AutoPoison":
        args.model_path = "TaiGary/AutoPoison"
        args.model_ref_path = "meta-llama/Llama-2-7b-chat-hf"
        model_pruned_path = "attribution_code/model/AutoPoison/unstructured/wanda_weightonly/align/"
        args.model_template = "alpaca"


        
        
    print(f"Output to {args.answer_file}")


    run_eval(
        model_path=args.model_path,
        model_ref_path = args.model_ref_path,
        model_template=args.model_template,
        question_file=args.question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=args.answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        defense=args.defense,
    )

    reorg_answer_file(args.answer_file)
