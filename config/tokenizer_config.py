"""
Tokenization 配置类
"""

import yaml
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class TokenizerConfig:
    # 分词器配置
    vocab_size: Optional[int] = None
    special_tokens_num: Optional[int] = None
    model_max_length: Optional[int] = None

    # 行为配置
    add_bos_token: Optional[bool] = None
    add_eos_token: Optional[bool] = None
    add_prefix_space: Optional[bool] = None
    clean_up_tokenization_spaces: Optional[bool] = None
    legacy: Optional[bool] = None
    sp_model_kwargs: Optional[Dict[str, Any]] = None
    spaces_between_special_tokens: Optional[bool] = None
    tokenizer_class: Optional[str] = None

    # 特殊token配置
    special_tokens: Optional[Dict[str, List[str]]] = None

    # 模型角色token映射
    role_tokens: Optional[Dict[str, str]] = None

    # 多模态token
    multi_modal: Optional[Dict[str, str]] = None

    # 对话模板
    chat_template: Optional[str] = None

    # 训练数据配置
    data: Optional[Dict[str, str]] = None

    # 输出配置
    output: Optional[Dict[str, str]] = None
    
    """docstring for TokenizerConfig."""
    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    
if __name__=="__main__":
    config = TokenizerConfig.from_yaml("/Users/adisihansun/Desktop/Git Repo/Practical/NanoLLM/config/tokenizer_config.yaml")
    print(config.special_tokens)
    print(type(config.special_tokens))