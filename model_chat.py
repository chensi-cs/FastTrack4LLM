#!/usr/bin/env python3
# -*- coding: utf-8 -*-  # 声明编码为 UTF-8
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from models.llama_model import Llama1ForCausalLM 
from utils.config import ChatConfig

def init_model(config,ischeckpoint=True):
    model = Llama1ForCausalLM(config)
    # strict=True（默认）：严格检查权重文件中的参数名与模型结构中的参数名是否完全匹配。若有不匹配（如缺少参数或多了无关参数），会抛出 RuntimeError。
    # 正确写法（提取模型权重）
    if ischeckpoint:
        # 加载模型权重文件
        print(f"加载checkpoint模型权重文件: {config.model_path}")
        checkpoint = torch.load(config.model_path, map_location=config.device)
        state_dict = checkpoint['model_state_dict']  # 只取模型权重部分

        # 加载到模型
        model.load_state_dict(state_dict, strict=True)
    else:
        print(f"加载最终模型权重文件: {config.model_path}")
        model = load_state_dict(torch.load(config.model_path, map_location=config.device), strict=True)


    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

    # 计算模型总参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total_params:.3f} M")  # 保留两位小数

    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'LLM可训练参数量: {trainable_params:.3f} M')

    return model, tokenizer

def prompt_template():
    prompts = [
        '大模型基本原理介绍',
        '湖南美食有什么',
        '中国有哪些旅游景点',
    ]
    return prompts


def main():
    config = ChatConfig()
    config.model_path = 'checkpoints/checkpoint_epoch_1.pt'  # 模型权重文件路径

    model, tokenizer = init_model(config,ischeckpoint=True)
    model.eval()
    model.to(config.device)
    prompt_templates = prompt_template()

    # TextStreamer 实现文本的流式输出，即生成的文本会像打字机一样逐个 token 实时显示，而非等完整结果生成后一次性输出。
    # skip_prompt=True,      # 不显示输入的提示文本
    # skip_special_tokens=True  # 不显示特殊token（如<|EndOfText|>）
    streamer = TextStreamer(tokenizer,skip_special_tokens=True )
    messages = []

    test_mode = int(input(" 请输入测试模式： [0] 自动测试  [1] 手动测试\n"))
    # iter(lambda: input('👶: '), ''）用于创建一个持续接收用户输入的迭代器，直到用户输入空字符串（直接按回车）才会停止。
    for idx ,prompt in enumerate(prompt_templates if test_mode==0 else iter( lambda: input("👶："), "")):
        messages.append({"role":"user","content":prompt})
        print(f'👶: {prompt}')
        new_prompt = tokenizer.bos_token + prompt

        input_ids =  tokenizer(
            str(new_prompt),
            truncation=True,
            return_tensors='pt'
        )
        input_ids = input_ids.input_ids
        input_ids = input_ids.to(config.device)  

        # do_sample=True ?
        output_ids = model.generate(
            input_ids=input_ids,                             # 输入的 Token 序列（模型生成的起点）
            max_new_tokens=config.max_generate_len,  # 最多生成的新 Token 数量
            num_return_sequences=1,                  # 生成的候选序列数量
            do_sample=True,                          # 启用采样策略（而非贪心搜索）
            streamer=streamer,                       # 流式输出器（实时显示生成过程）
            temperature=config.temperature,          # 温度参数（控制生成随机性）
            top_p=config.top_p,                      # 核采样参数（控制生成多样性）
            pad_token_id=tokenizer.pad_token_id,     # 填充 Token 的 ID
            eos_token_id=tokenizer.eos_token_id,     # 结束 Token 的 ID
            bos_token_id=tokenizer.bos_token_id     # 开始 Token 的 ID
            # attention_mask=input_ids.attention_mask  # 注意力掩码（标记有效 Token）
        )

        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:],skip_special_tokens=True)
        messages.append({"role":"assistant","content":response})
        print(f'🤖️: {response}')
        print("----------------------------")

    print("对话历史：")
    for message in messages:
        print(message)
        
if __name__ == '__main__':
    main()
