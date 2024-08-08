# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire

from llama import Dialog, Llama


def generate_user_prompt():
    user_input = input().strip().lower()
    return {"role": "user", "content": user_input}

def parse_tool_call(response):
    eom_id = "<|reserved_special_token_4|>"
    python_tag = "<|reserved_special_token_5|>"
    response = response.replace(python_tag, "*1")
    response = response.replace(eom_id, "*1")
    return response.split("*1")[1]

def get_data_from_tool(tool_call):
    return '{"queryresult": {"success": true, "inputstring": "solve x^3 - 4x^2 + 6x - 24 = 0", "pods": [{"title": "Input interpretation", "subpods": [{"title": "", "plaintext": "solve x^3 - 4 x^2 + 6 x - 24 = 0"}]}, {"title": "Results", "primary": true, "subpods": [{"title": "", "plaintext": "x = 4"}, {"title": "", "plaintext": "x = \u00b1 (i sqrt(6))"}]}, ... ]}}'

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    system_prompt = {"role": "system", "content": "Environment: ipython\nTools: brave_search, wolfram_alpha\n\nCutting Knowledge Date: December 2023\nToday Date: 08 Aug 2024\nYou are a helpful assistant."}
    dialog: Dialog = [system_prompt]
    try:
        while True:
            # Get user input
            print("User prompt >>")
            user_input = generate_user_prompt()
            dialog.append(user_input)
            
            # Convert text to tokens for the base model
            # tokens = generator.tokenizer.encode(user_input, bos=True, eos=False)
            tokens = generator.formatter.encode_dialog_prompt(dialog)
            
            # Print the text version of the encoded dialog
            # print(generator.tokenizer.decode(tokens))

            # Get return tokens
            if not max_gen_len:
                max_gen_len = 1000
            output_tokens, _ = generator.generate(
                [tokens],
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p
            )
            response = generator.tokenizer.decode(output_tokens[0])
            print(response)
            if response.startswith("<|reserved_special_token_5|>"):
                tool_call = parse_tool_call(response)
                print(f"{tool_call=}")
                dialog.append({"role": "assistant", "content": "<|reserved_special_token_5|>"+tool_call+"<|reserved_special_token_4|>"})
                data_from_tool = get_data_from_tool(tool_call)
                dialog.append({"role": "ipython", "content": data_from_tool})
                tokens = generator.formatter.encode_dialog_prompt(dialog)
                output_tokens, _ = generator.generate(
                    [tokens],
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p
                )
                response = generator.tokenizer.decode(output_tokens[0])
            dialog.append({"role": "assistant", "content": response})
            print(response)
            print("------------------------------------------------")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    fire.Fire(main)
