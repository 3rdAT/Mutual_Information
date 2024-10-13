from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import fire
import torch
import torch.nn.functional as F
from src.hf_utils import init_hf


def main(model_name: str="meta-llama/Meta-Llama-3-8B-Instruct"):

    model, tokenizer = init_hf(model_name)

    prompted = '''Please exactly repeat the following characters and strictly don't include anything else: With great power comes great repsanility.'''
    







if __name__ == "__main__":
    fire.Fire(main)