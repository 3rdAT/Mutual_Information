from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import fire
import torch
import torch.nn.functional as F

def main(
    model_name: str="meta-llama/Meta-Llama-3-8B",
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token='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', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        token='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',
    )

    model.eval()

    #Basically, in the prompted string I need to add the answer as well twice. I need to make sure that I append the appropriate token ids in the answer,
    #To ensure that I need to have the same token_ids in the prompted's two parts and the target's part.

    prompted = '''Please exactly repeat the following characters and strictly don't include anything else: With great power comes great repsanility.''' #Three tokens
    target = '''abcde'''

    tokenizer.chat_template = ("{% if messages[0]['role'] == 'system' %}"
                                    "{% set offset = 1 %}"
                                "{% else %}"
                                    "{% set offset = 0 %}"
                                "{% endif %}"

                                "{{ bos_token }}"
                                "{% for message in messages %}"
                                    "{% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}"
                                        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
                                    "{% endif %}"

                                    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
                                "{% endfor %}"

                                "{% if add_generation_prompt %}"
                                    "{{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}"
                                "{% endif %}"
                                )

    # tokenizer.chat_template = ("{% if messages[0]['role'] == 'system' %}"
    #                                 "{% set system_message = '<<SYS>>\n' + messages[0]['content'] | trim + '\n<</SYS>>\n\n' %}"
    #                                 "{% set messages = messages[1:] %}"
    #                             "{% else %}"
    #                                 "{% set system_message = '' %}"
    #                             "{% endif %}"

    #                             "{% for message in messages %}"
    #                                 "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    #                                     "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    #                                 "{% endif %}"

    #                                 "{% if loop.index0 == 0 %}"
    #                                     "{% set content = system_message + message['content'] %}"
    #                                 "{% else %}"
    #                                     "{% set content = message['content'] %}"
    #                                 "{% endif %}"

    #                                 "{% if message['role'] == 'user' %}"
    #                                     "{{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}"
    #                                 "{% elif message['role'] == 'assistant' %}"
    #                                     "{{ ' ' + content | trim + ' ' + eos_token }}"
    #                                 "{% endif %}"
    #                             "{% endfor %}")

    messages = [
        {
            "role":"user",
            "content":f"{prompted}"
        },
        # {
        #     "role":"assistant",
        #     "content":"abcdefghijklmnopqrstuvwx"
        # }
    ]
    

    # target_tokens = tokenizer(target, return_tensors="pt").input_ids

    batch = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"The templated prompt:\n{batch}")
    batch = batch + "With great power comes great resp."

    batch = tokenizer(batch, return_tensors="pt", add_special_tokens=False)
    prompt_len = len(batch['input_ids'][0])
    print(f"The length is: {prompt_len}")
    batch = {k: v.to("cuda") for k, v in batch.items()}


    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=4096, #restricts the number new tokens to be generated. I have kept it to the len(target_prompt_tokens)-1 (1 coz of having a <s> token)
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

    print("The model's output is:")
    print(tokenizer.decode(outputs.sequences[0]))
    print("=====================================")
    print("The token is:",outputs.sequences[0][-3:][0])
    print("The response is:", tokenizer.decode(outputs.sequences[0][-3:][0]))
    output_text = tokenizer.decode(outputs.sequences[0][-3:][0])

    # print("Sanity on token_id",tokenizer.decode([12014]))


    # Convert logits to log probabilities
    log_probs = []
    for score in outputs.scores:
        # Apply softmax to get probabilities, then take the log
        probs = F.softmax(score, dim=-1)
        log_prob = torch.log(probs)
        log_probs.append(log_prob)


    print("The model's output is:\n",output_text)
    print(len(log_probs))

    log_prob_last_position = log_probs[-1][0]  # Accessing the second-to-last logits and the first token's logits
    top_k = 10  # Set top_k to your desired value

    # Get top_k log probabilities and their indices
    top_k_probs, top_k_indices = torch.topk(log_prob_last_position, top_k, dim=-1)

    print(f"The prompt is:{prompted}")

    print(f"Top {top_k} probabilities:\n", top_k_probs)
    print(f"Top {top_k} indices:\n", top_k_indices)
    print(f"Top {top_k} tokens:\n", tokenizer.decode(top_k_indices))

    # Loop through the top_k indices and decode each token
    for i, token_id in enumerate(top_k_indices):
        decoded_token = tokenizer.decode([token_id.item()])  # Decode the token ID
        prob = top_k_probs[i].item()  # Get the corresponding log probability
        print(f"Token {i+1}: {decoded_token}, Probability: {prob}")

if __name__ == "__main__":
    fire.Fire(main)