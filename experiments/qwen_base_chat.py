import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # 加载模型和分词器
    print("正在加载 Qwen2.5-7B 模型和分词器...")
    model_name = "Qwen/Qwen2.5-7B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print(f"模型 {model_name} 已加载完成！")
    print("这是基础模型，没有经过指令微调，所以回复可能不符合预期。")
    print("输入 'exit' 退出对话")
    
    # 打印特殊tokens
    print("Special tokens:")
    print(f"EOS token: {tokenizer.eos_token}, id: {tokenizer.eos_token_id}")
    print(f"BOS token: {tokenizer.bos_token}, id: {tokenizer.bos_token_id}")
    print(f"PAD token: {tokenizer.pad_token}, id: {tokenizer.pad_token_id}")
    print(f"UNK token: {tokenizer.unk_token}, id: {tokenizer.unk_token_id}")

    # 查看词表大小
    print(f"\nVocabulary size: {len(tokenizer)}")
    
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == 'exit':
            print("感谢使用，再见！")
            break
        
        # 生成回复
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n模型: {response}")

if __name__ == "__main__":
    main()
    
# (opencompass_lmdeploy_118) root@di-20241121034131-v8bjg /fs-computility/mllm1/limo/workspace/stepfun/Open-Reasoner-Zero ➜ git:(main) python experiments/qwen.py
# 正在加载 Qwen2.5-7B 模型和分词器...
# Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:07<00:00,  1.77s/it]
# 模型 Qwen/Qwen2.5-7B 已加载完成！
# 这是基础模型，没有经过指令微调，所以回复可能不符合预期。
# 输入 'exit' 退出对话
# Special tokens:
# EOS token: <|endoftext|>, id: 151643
# BOS token: None, id: None
# PAD token: <|endoftext|>, id: 151643
# UNK token: None, id: None

# Vocabulary size: 151665

# 用户: A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.\nThis is the problem:\nWhat is the greatest integer value of $c$ such that $-9$ is not in the range of $y = x^2 + cx + 20$?\nAssistant: <think>
# Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.

# 模型:  To find the greatest integer value of c such that -9 is not in the range of y = x^2 + cx + 20, we need to determine the maximum value of c that keeps -9 from being a value of y. 

# First, let's rewrite the equation as y = x^2 + cx + 20 - 9, which simplifies to y = x^2 + cx + 11.

# Next, we want to find the maximum value of c such that the quadratic equation x^2 + cx + 11 = 0 has no real solutions. This means that the discriminant of the quadratic equation must be less than 0.

# The discriminant of a quadratic equation ax^2 + bx + c = 0 is given by b^2 - 4ac. In our case, a = 1, b = c, and c = 11. So the discriminant is c^2 - 4(1)(11).

# We want c^2 - 44 < 0. Solving for c, we get c^2 < 44. Taking the square root of both sides, we get c < sqrt(44).

# Now, we need to find the greatest integer value of c that satisfies this inequality. sqrt(44) is approximately 6.63, so the greatest integer value of c is 6.

# Therefore, the greatest integer value of c such that -9 is not in the range of y = x^2 + cx + 20 is 6.</think> <answer> 6 </answer>

# 用户: 


