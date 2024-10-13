
def chatty(model_name, tokenizer):

    if model_name == "llama3":
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
        
    elif model_name == "llama2":
        tokenizer.chat_template = ("{% if messages[0]['role'] == 'system' %}"
                                "{% set system_message = '<<SYS>>\n' + messages[0]['content'] | trim + '\n<</SYS>>\n\n' %}"
                                "{% set messages = messages[1:] %}"
                            "{% else %}"
                                "{% set system_message = '' %}"
                            "{% endif %}"

                            "{% for message in messages %}"
                                "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
                                    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
                                "{% endif %}"

                                "{% if loop.index0 == 0 %}"
                                    "{% set content = system_message + message['content'] %}"
                                "{% else %}"
                                    "{% set content = message['content'] %}"
                                "{% endif %}"

                                "{% if message['role'] == 'user' %}"
                                    "{{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}"
                                "{% elif message['role'] == 'assistant' %}"
                                    "{{ ' ' + content | trim + ' ' + eos_token }}"
                                "{% endif %}"
                            "{% endfor %}")
        
    elif model_name == "mistral":
        tokenizer.chat_template = ("{% if messages[0]['role'] == 'system' %}"
                                        "{% set system_message = messages[0]['content'] | trim + '\n\n' %}"
                                        "{% set messages = messages[1:] %}"
                                    "{% else %}"
                                        "{% set system_message = '' %}"
                                    "{% endif %}"

                                    "{{ bos_token + system_message}}"
                                    "{% for message in messages %}"
                                        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
                                            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
                                        "{% endif %}"

                                        "{% if message['role'] == 'user' %}"
                                            "{{ '[INST] ' + message['content'] | trim + ' [/INST]' }}"
                                        "{% elif message['role'] == 'assistant' %}"
                                            "{{ ' ' + message['content'] | trim + eos_token }}"
                                        "{% endif %}"
                                    "{% endfor %}")
        
    return tokenizer