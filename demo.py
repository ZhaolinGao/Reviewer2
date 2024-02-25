import sys
import json
import math
import torch
import argparse
import transformers
from llama_attn_replace import replace_llama_attn

SEQ_LEN = 32768


def parse_args():
    parser = argparse.ArgumentParser(description="Reviewer2 Demo")
    parser.add_argument('--json_path', type=str, default='', help="path to paper json file")
    return parser.parse_args()


def build_generator(
    model, tokenizer, temperature=0.7, top_p=0.7, top_k=50, max_new_tokens=1024, min_new_tokens=64, repetition_penalty=1.13
):
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        output = model.generate(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=True,
        )
        
        out = tokenizer.decode(output[0], skip_special_tokens=True)

        try:
            out = out.split(prompt.lstrip("<s>"))[1].strip()
        except:
            out = []

        return out

    return response


if __name__ == '__main__':

    args = parse_args()

    # prep model
    replace_llama_attn(inference=True)

    # ==========Prompt Model==========

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained('GitBag/Reviewer2_Mp')
    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}
    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if SEQ_LEN > orig_ctx_len:
            scaling_factor = float(math.ceil(SEQ_LEN / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            print(f'rope scaling factor {scaling_factor}')

    # prep model
    prompt_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'GitBag/Reviewer2_Mp',
        model_max_length=SEQ_LEN if SEQ_LEN > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )

    prompt_model = transformers.AutoModelForCausalLM.from_pretrained(
        'GitBag/Reviewer2_Mp',
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    prompt_model.resize_token_embeddings(32001)
    prompt_model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        prompt_model = torch.compile(prompt_model)

    # ==========Paper Content==========

    with open(args.json_path, 'rb') as p:
        paper_data = json.load(p)

    paper_content = []
    paper_content.append('Title')
    paper_content.append(paper_data['metadata']['title'])
    paper_content.append('Abstract')
    paper_content.append(paper_data['metadata']['abstractText'])
            
    # use full paper
    for section in paper_data['metadata']['sections']:
        paper_content.append(section['heading'])
        paper_content.append(section['text'])
    for i in range(len(paper_content)):
        if paper_content[i] == None:
            paper_content[i] = 'N/A'
    paper_content = "\n".join(paper_content).encode("utf-8", "ignore").decode("utf-8").strip()

    # ==========Prompt Generation==========

    prompt_Llama_2 = (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>>\nRead the following paper carefully:\n{paper_content}\n\n\n"
        "Your task is to construct a list of questions about the paper for the reviewer to answer.\n"
        "\nThe reviewer should answer in the following format:\n{format}\n"
        "[/INST]"
    )
    prompt_dict = {
        'paper_content': paper_content,
        'format': '\n'.join(['Summary Of The Paper', 'Strengths And Weaknesses'])
    }
    prompt = prompt_Llama_2.format_map(prompt_dict)

    prompt_generator = build_generator(prompt_model, prompt_tokenizer)
    gen_prompt = prompt_generator(prompt)
    print("Generated prompt\n", gen_prompt)

    # ==========Review Model==========
    del prompt_model

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained('GitBag/Reviewer2_Mr')
    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}
    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if SEQ_LEN > orig_ctx_len:
            scaling_factor = float(math.ceil(SEQ_LEN / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            print(f'rope scaling factor {scaling_factor}')

    # prep model      
    reviewer_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'GitBag/Reviewer2_Mr',
        model_max_length=SEQ_LEN if SEQ_LEN > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )

    review_model = transformers.AutoModelForCausalLM.from_pretrained(
        'GitBag/Reviewer2_Mr',
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    review_model.resize_token_embeddings(32001)
    review_model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        review_model = torch.compile(review_model)

    # ==========Review Generation==========

    prompt_Llama_2 = (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>>\nRead the following paper carefully:\n{paper_content}\n\n\n"
        "Your task is to compose a high-quality review of the paper submitted to a top-tier conference.\n"
        "Your review should contain the answers to the following questions:\n{prompt_gen}\n"
        "\nWrite your review into following section:\n{format}\n"
        "[/INST]"
    )
    prompt_dict = {
        'paper_content': paper_content,
        'prompt_gen': gen_prompt,
        'format': '\n'.join(['Summary Of The Paper', 'Strengths And Weaknesses', 'Questions', 'Limitations'])
    }
    prompt = prompt_Llama_2.format_map(prompt_dict)

    review_generator = build_generator(review_model, reviewer_tokenizer)
    gen_review = review_generator(prompt)
    print("Generated revew\n", gen_review)