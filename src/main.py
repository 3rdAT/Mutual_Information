from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import fire
import torch
import torch.nn.functional as F
from hf_utils import init_hf
from chat_templates import chatty


def main(model_name: str="meta-llama/Meta-Llama-3-8B-Instruct"):

    model, tokenizer = init_hf(model_name)

    prompted = '''Please exactly repeat the following characters and strictly don't include anything else: With great power comes great repsanility.'''

    tokenizer = chatty('llama3',tokenizer)

    messages = [
        {
            "role":"user",
            "content":f"{prompted}"
        },
    ]

    batch = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch = tokenizer(batch, return_tensors="pt", add_special_tokens=False)
    prompt_len = len(batch['input_ids'][0])
    print(prompt_len)
    batch = {k: v.to("cuda") for k, v in batch.items()}

    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=4096, #restricts the number new tokens to be generated. I have kept it to the len(target_prompt_tokens)-1 (1 coz of having a <s> token)
            do_sample=False,
            top_p=1.0,
            temperature=1.0,
            min_length=None,
            use_cache=True,
            top_k=50,
            repetition_penalty=1.0,
            length_penalty=1,
            output_hidden_states= True, return_dict_in_generate=True,
            output_scores=True,
        )
    output_tokens = outputs.sequences[0][prompt_len:-1]
    logits_p = outputs.scores[:-1]
    output_text = tokenizer.decode(output_tokens)

    print(output_text)

    start_token = torch.tensor([tokenizer.bos_token_id], device=model.device)
    teacher_forced_input_ids = torch.cat((start_token, output_tokens))
    teacher_attention_mask = torch.ones(teacher_forced_input_ids.shape, device=teacher_forced_input_ids.device)

    batch_t = {'input_ids':teacher_forced_input_ids, 'attention_mask':teacher_attention_mask}
    batch_t = {k: v.to("cuda") for k, v in batch_t.items()}

    #Lets do a forward pass:
    outputs_t = model(**batch_t)

    print(outputs_t.keys)



if __name__ == "__main__":
    fire.Fire(main)