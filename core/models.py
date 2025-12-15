"""
模型加载与管理 - 使用官方CLIP库
"""
import os
import torch
from typing import Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from config import MODEL_CONFIG, load_api_keys

# 设置环境变量，使用HuggingFace镜像（如果需要下载模型）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class ModelManager:
    """模型管理器"""

    def __init__(self, use_api: bool = False):
        self.use_api = use_api
        self.api_keys = load_api_keys()
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")

        # 设置CLIP缓存目录为之前下载的目录
        self.clip_cache_dir = os.path.expanduser("~/.cache/clip")

        # 确保目录存在
        os.makedirs(self.clip_cache_dir, exist_ok=True)
        print(f"CLIP缓存目录: {self.clip_cache_dir}")

    def get_text_embedder(self):
        """获取文本嵌入模型"""
        if "text_embedder" not in self.models:
            config = MODEL_CONFIG["text_embedding"]
            model_name = config["name"]

            print(f"加载文本嵌入模型: {model_name}")
            model = SentenceTransformer(model_name)
            model.to(self.device)
            self.models["text_embedder"] = model

        return self.models["text_embedder"]

    def get_image_embedder(self):
        """获取图像嵌入模型 - 使用官方CLIP库"""
        if "clip_model" not in self.models:
            print("加载CLIP模型...")

            try:
                # 导入openai的CLIP库
                import clip
            except ImportError:
                print("未安装openai/clip库，请运行: pip install git+https://github.com/openai/CLIP.git")
                raise

            try:
                # 加载之前下载过的CLIP模型 ViT-B/32
                model_name = "ViT-B/32"

                # 设置环境变量，指定缓存目录
                os.environ["CLIP_CACHE_DIR"] = self.clip_cache_dir

                # 加载模型，指定缓存目录
                model, preprocess = clip.load(model_name,
                                              device=self.device,
                                              download_root=self.clip_cache_dir)
                model.eval()

                self.models["clip_model"] = model
                self.models["clip_processor"] = preprocess

                print(f"CLIP模型 {model_name} 加载完成，来自: {self.clip_cache_dir}")

            except Exception as e:
                print(f"加载CLIP模型失败: {e}")
                print("尝试使用备用模型...")

                # 备用方案：使用Transformers库的CLIP
                try:
                    from transformers import CLIPModel, CLIPProcessor
                    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    model.to(self.device)
                    model.eval()

                    self.models["clip_model"] = model
                    self.models["clip_processor"] = processor

                    print("备用CLIP模型加载完成")
                except Exception as e2:
                    print(f"备用模型也加载失败: {e2}")
                    raise

        return self.models["clip_model"], self.models["clip_processor"]

    def get_text_embeddings(self, texts: list) -> np.ndarray:
        """获取文本嵌入"""
        if self.use_api and self.api_keys.get("gemini"):
            try:
                return self._get_embeddings_via_api(texts)
            except Exception as e:
                print(f"API调用失败: {e}，使用本地模型")

        # 使用本地模型
        model = self.get_text_embedder()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """获取图像嵌入 - 使用官方CLIP库"""
        try:
            # 获取CLIP模型和预处理函数
            clip_model, preprocess = self.get_image_embedder()

            # 加载并预处理图像
            image = Image.open(image_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # 官方CLIP库使用encode_image方法
                image_features = clip_model.encode_image(image_input)

            # 归一化
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten()

        except Exception as e:
            print(f"处理图像失败 {image_path}: {e}")
            # 返回随机向量作为fallback
            return np.random.randn(512)

    def get_query_embedding(self, query: str) -> np.ndarray:
        """获取查询文本的CLIP嵌入（用于图像搜索）"""
        clip_model, preprocess = self.get_image_embedder()

        # 使用官方CLIP库处理文本
        import clip  # 确保已导入

        # 将文本转换为token
        text = clip.tokenize([query]).to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(text)

        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()

    # ... 其余方法保持不变 ...

    def classify_paper(self, text: str, categories: list = None) -> str:
        """使用嵌入相似度分类论文"""
        from config import PAPER_CATEGORIES

        if categories is None:
            categories = PAPER_CATEGORIES

        # 计算文本与每个类别的相似度
        text_embedding = self.get_text_embeddings([text])[0]
        category_embeddings = self.get_text_embeddings(categories)

        # 计算余弦相似度
        similarities = np.dot(category_embeddings, text_embedding) / (
                np.linalg.norm(category_embeddings, axis=1) * np.linalg.norm(text_embedding)
        )

        # 返回最相似的类别
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        # 如果相似度太低，使用关键词匹配
        if best_score < 0.4:  # 降低阈值，因为模型不够准确
            return self.classify_by_keywords(text, categories)

        return categories[best_idx]

    def classify_by_keywords(self, text: str, categories: list) -> str:
        """使用关键词匹配分类（作为备用）"""
        text_lower = text.lower()

        # 增强的关键词映射 - 专门针对你的论文类型
        keywords = {
            "Computer Vision": [
                # 通用CV关键词
                "vision", "image", "cnn", "convolutional", "object detection",
                "segmentation", "recognition", "face", "video", "optical flow",
                "feature extraction", "pose estimation", "3d reconstruction",
                # 深度伪造相关关键词
                "deepfake", "artifact", "deceptive", "compressed", "forgery",
                "fake detection", "manipulation detection", "face swapping",
                "generative adversarial", "gan", "synthetic media",
                "social network", "online platform", "media forensics"
            ],
            "Natural Language Processing": [
                "language", "text", "transformer", "attention", "nlp", "bert",
                "gpt", "token", "embedding", "sentiment", "translation"
            ],
            "Reinforcement Learning": [
                "reinforcement", "rl", "q-learning", "policy", "agent",
                "environment", "reward", "markov", "monte carlo"
            ],
            "Machine Learning": [
                "machine learning", "regression", "classification", "clustering",
                "svm", "random forest", "decision tree", "knn"
            ],
            "Deep Learning": [
                "deep learning", "neural network", "backpropagation", "gradient",
                "activation", "layer", "training", "inference"
            ],
            "Security": [
                "security", "attack", "defense", "malware", "robust", "detection",
                "adversarial", "privacy", "encryption", "authentication"
            ]
        }

        scores = {}
        for category, words in keywords.items():
            if category not in categories:
                continue
            score = sum(1 for word in words if word in text_lower)
            scores[category] = score

        if scores:
            best_category = max(scores, key=scores.get)
            if scores[best_category] > 0:
                print(f"关键词分类: {best_category} (分数: {scores[best_category]})")
                return best_category

        return "Uncategorized"