import argparse
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM
import os.path as osp

def main():
    parser = argparse.ArgumentParser(description='Convert LLaMA model with LoRA/ALPaCAC layers to a standard HuggingFace checkpoint.')
    parser.add_argument('--base_model', type=str, default="jeffwan/vicuna-13b", help='Base model name, e.g. "jeffwan/vicuna-13b"')
    parser.add_argument('--model_folder', type=str, required=True, help='Output model folder path')

    args = parser.parse_args()
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    lora_model = PeftModel.from_pretrained(
        base_model,
        args.model_folder,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    for layer in lora_model.base_model.model.model.layers:
        layer.self_attn.q_proj.merge_weights = True
        layer.self_attn.v_proj.merge_weights = True

    lora_model.train(False)

    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }

    LlamaForCausalLM.save_pretrained(
        base_model, osp.join(args.model_folder, "hf_checkpoint"), state_dict=deloreanized_sd, max_shard_size="800MB"
    )

if __name__ == "__main__":
    main()