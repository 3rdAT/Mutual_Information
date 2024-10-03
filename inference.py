from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import fire

def main(
    model_name: str="meta-llama/Llama-2-7b-hf",
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens = 3, #The maximum numbers of tokens to generate
    output_file: str="/data/data/arrv/Think/eval/it2/l2-sft(1).json",
    seed: int=42, #seed value for reproducibility
    do_sample: bool=False, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token='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', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-hf',
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        token='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',
    )

    model.eval()

    user_prompt = '''Please exactly copy the following character sequence: abcde''' #Three tokens
    target = '''abcde''' #Three tokens
    target_tokens = tokenizer(target, return_tensors="pt").input_ids
    batch = tokenizer(user_prompt, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}

    #SANITY_CHECK_ON_TARGET_IN_USER_PROMPT
    print("The length of the target_tokens are",len(target_tokens[0]))
    print(target_tokens[0])
    print("The target in the prompt is: ", tokenizer.decode(batch['input_ids'][0, -len(target_tokens[0])+1:])) #we are doing plus 1 to ignore the <s> token
    print(batch['input_ids'])
    
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
    
    log_probs = torch.log_softmax(logits, dim=-1)

    target_log_probs = log_probs[0, -len(target_tokens[0]):-1] #Log probs in the prompt. The shape is [len(target_tokens), vocab_size]


    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=len(target_tokens[0])-1, #restricts the number new tokens to be generated. I have kept it to the len(target_prompt_tokens)-1 (1 coz of having a <s> token)
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            min_length=min_length,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            output_hidden_states= True, return_dict_in_generate=True,
            output_scores=True,
            **kwargs
        )
    output_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    print("The model's output is:",output_text[len(user_prompt):])

    logits = outputs.scores

    log_probs_answer = torch.stack([torch.log_softmax(logit, dim=-1) for logit in logits], dim=0)

    log_probs_answer = log_probs_answer.squeeze(1)

    print("The shape of answer log probs:",log_probs_answer.shape)
    print("The shape of target log probs:",target_log_probs.shape)

    print(target_log_probs)
    print(log_probs_answer)

if __name__ == "__main__":
    fire.Fire(main)
