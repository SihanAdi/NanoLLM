""" 
Byte Pair Encoding 字节对编码
"""

import os
import sys
import json
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.tokenizer_config import TokenizerConfig
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

def get_texts(conf: TokenizerConfig):
    with open(conf.data["path"], "r", encoding=conf.data["encoding"], errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= 10000:
                break
            else:
                try:
                    data = json.loads(line)
                    contents = [item.get('content') for item in data.get('conversations', []) if item.get('content')]
                    if contents:
                        yield "\n".join(contents)
                except:
                    continue

def train_tokenizer(conf: TokenizerConfig):
    """训练 tokenizer"""
    # 1. 初始化一个基于 BPE 模型的 Tokenizer
    tokenizer = Tokenizer(BPE())
    
    # 2. 配置预分词器（使用经典的字节级预分词，自动包含正则切分）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # type: ignore
    
    # 3. 设置 special_tokens
    special_tokens = conf.special_tokens
    special_token_list = special_tokens["base"] if special_tokens else []
    additional_token_list = special_tokens["tool"] if special_tokens else []
    # 预留一定数量的token位置, 方便未来扩展功能而不需要重新训练分词器
    num_buffer = conf.special_tokens_num - len(special_token_list) - len(additional_token_list)
    buffer_token_list = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)]
    all_special_tokens = special_token_list + additional_token_list + buffer_token_list
    
    # 4. 设置训练器，指定目标词表大小
    trainer = BpeTrainer(
        vocab_size=conf.vocab_size,
        min_frequency=2, # 最小频次阈值，低于该频次不合并
        show_progress=True, # 显示进度条
        special_tokens=all_special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet() # 初始化字母表（所有字节）
    )
    
    # 5. 加载训练文本
    texts = get_texts(conf)
    
    # 6. 训练 + 设置解码器
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(special_token_list)
    
    # 7. 保存分词器
    tokenizer_dir = conf.output["dir"]
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    tokenizer.save(tokenizer_json_path)
    tokenizer.model.save(tokenizer_dir)
    
    # 8. 后处理 + 设置配置文件
    # with open(tokenizer_json_path, "r", encoding="utf-8") as f:
    #     tokenizer_data = json.load(f)
    # # 将 additional 和 buffer token 的 special 属性设为 false
    # for token_info in tokenizer_data.get("added_tokens", []):
    #     if token_info["content"] not in special_token_list:
    #         token_info["special"] = False
    # with open(tokenizer_json_path, "w", encoding="utf-8") as f:
    #     json.dumps(tokenizer_data, f, ensure_ascii=False, indent=2)
        
    


if __name__=="__main__":
    config_path = project_root / "config" / "tokenizer_config.yaml"
    config = TokenizerConfig.from_yaml(str(config_path))
    
    train_tokenizer(config)

