"""
配置文件 - 更新模型配置
"""
import os
import json
from pathlib import Path
from typing import Dict, Any

# 项目根目录
BASE_DIR = Path(__file__).parent.absolute()

# 数据目录
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
IMAGES_DIR = DATA_DIR / "images"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# 模型配置 - 简化版本
MODEL_CONFIG = {
    # 文本嵌入模型 (轻量级，适合CPU)
     "text_embedding": {
        "name": "paraphrase-albert-small-v2",  # 更小，更容易下载
        "provider": "sentence_transformers",
        "cache_dir": str(BASE_DIR / "models"),
        "device": "cpu"
    },

    # 图像嵌入模型 - 使用transformers中的CLIP
    "image_embedding": {
        "name": "openai/clip-vit-base-patch32",
        "provider": "transformers",
        "cache_dir": str(BASE_DIR / "models"),
        "device": "cpu"
    }
}

# 分类配置
PAPER_CATEGORIES = [
    "Computer Vision",
    "Natural Language Processing",
    "Reinforcement Learning",
    "Machine Learning",
    "Deep Learning",
    "AI Theory",
    "Robotics",
    "Multimodal",
    "Uncategorized"
]

# 向量数据库配置
VECTOR_DB_CONFIG = {
    "text_collection": "papers",
    "image_collection": "images",
    "text_embedding_dim": 384,  # all-MiniLM-L6-v2的维度
    "image_embedding_dim": 512,  # CLIP的维度
    "persist_directory": str(VECTOR_DB_DIR)
}

def load_api_keys() -> Dict[str, str]:
    """加载API密钥"""
    keys_path = BASE_DIR / "api_keys.json"
    if keys_path.exists():
        with open(keys_path, 'r') as f:
            return json.load(f)
    return {}

def save_api_keys(**kwargs):
    """保存API密钥"""
    keys_path = BASE_DIR / "api_keys.json"
    existing = load_api_keys()
    existing.update(kwargs)

    with open(keys_path, 'w') as f:
        json.dump(existing, f, indent=2)

# 创建必要的目录
for dir_path in [DATA_DIR, PAPERS_DIR, IMAGES_DIR, VECTOR_DB_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# 为每个论文分类创建子目录
for category in PAPER_CATEGORIES:
    category_dir = PAPERS_DIR / category.replace(" ", "_")
    category_dir.mkdir(exist_ok=True)