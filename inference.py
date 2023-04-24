import pickle as pkl
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import torch
from utils import conv_v1_2, SeparatorStyle
from utils import generate_stream as generate_stream_func
import argparse
import os.path as osp

def args_parse():
    parser = argparse.ArgumentParser(description='Inference with AlpacaAPI')
    parser.add_argument('--model_folder', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda")
    return parser.parse_args()

class SimpleChatIO:
    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream, skip_echo_len: int):
        pre = 0
        for outputs in output_stream:
            outputs = outputs[skip_echo_len:].strip()
            outputs = outputs.split(" ")
            now = len(outputs) - 1
            if now > pre:
                print(" ".join(outputs[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(outputs[pre:]), flush=True)
        return " ".join(outputs)

def vicuna_chat(model_name, device, num_gpus, load_8bit=False, debug=False):
    prefix = """Below is an instruction that describes a task, paired with an API references that provides further about the API. Write code that appropriately completes the request.\n\n### Instruction:\n """
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: "13GiB" for i in range(num_gpus)},
                })
                
    model = LlamaForCausalLM.from_pretrained(model_name,
             load_in_8bit=load_8bit, low_cpu_mem_usage=True, **kwargs)
    tokenizer = LlamaTokenizer.from_pretrained("jeffwan/vicuna-13b", use_fast=False)
    chatio = SimpleChatIO()
    if device == "cuda" and num_gpus == 1:
        model.to(device)
    if debug:
        print(model)
        
    conv = conv_v1_2.copy()
    while True:
        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break
        with open("assets/vectorstore.pkl", "rb") as f:
            vectorstore = pkl.load(f)
        docs = vectorstore.similarity_search(inp, k=1)[0].page_content
        inp = prefix + inp + "\n\n### Input:\n" + docs + "\n\n### Code:"
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        skip_echo_len = len(prompt.replace("</s>", " ")) + 1

        params = {
            "model": model_name,
            "prompt": prompt,
            "temperature": 0.7,
            "max_new_tokens": 700,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(model, tokenizer, params, device)
        outputs = chatio.stream_output(output_stream, skip_echo_len)
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = outputs.strip()

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    
if __name__ == "__main__":
    args = args_parse()
    vicuna_chat(osp.join(args.model_folder ,"hf_checkpoint"), args.device, "auto")
