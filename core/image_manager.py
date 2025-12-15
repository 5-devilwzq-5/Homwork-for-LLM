"""
图像管理模块 - 更新为使用Transformers CLIP
"""
import os
import shutil
from pathlib import Path
from typing import List, Tuple
import chromadb
from chromadb.utils import embedding_functions
import numpy as np

from config import IMAGES_DIR, VECTOR_DB_CONFIG, BASE_DIR
from core.models import ModelManager

class ImageManager:
    """图像管理器"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.vector_db = self._init_vector_db()

    def _init_vector_db(self):
        """初始化向量数据库"""
        chroma_client = chromadb.PersistentClient(
            path=VECTOR_DB_CONFIG["persist_directory"]
        )

        # 创建或获取集合
        try:
            # 对于图像，我们使用自定义嵌入函数
            collection = chroma_client.get_collection(
                name=VECTOR_DB_CONFIG["image_collection"]
            )
        except:
            collection = chroma_client.create_collection(
                name=VECTOR_DB_CONFIG["image_collection"]
            )

        return collection

    def index_image(self, image_path: str) -> bool:
        """索引单张图像，并复制到项目images目录"""
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"文件不存在: {image_path}")
                return False

            # 检查文件大小
            if os.path.getsize(image_path) > 50 * 1024 * 1024:  # 50MB限制
                print(f"文件太大: {image_path}")
                return False

            # 生成唯一ID和存储路径
            import hashlib
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            # 获取文件扩展名
            file_ext = Path(image_path).suffix.lower()

            # 生成新文件名：哈希值 + 扩展名
            new_filename = f"{file_hash}{file_ext}"

            # 目标路径：data/images/哈希值.扩展名
            target_path = IMAGES_DIR / new_filename

            # 检查是否已存在
            existing = self.vector_db.get(ids=[file_hash])
            if existing['ids']:
                print(f"图像已存在: {image_path}")
                return True

            # 复制文件到项目目录
            try:
                shutil.copy2(image_path, target_path)
                print(f"已复制图像到: {target_path}")
            except Exception as e:
                print(f"复制文件失败: {e}")
                # 如果复制失败，仍然使用原始路径
                target_path = image_path

            # 获取图像嵌入
            embedding = self.model_manager.get_image_embedding(image_path)

            # 添加到向量数据库
            self.vector_db.add(
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "path": str(target_path),  # 存储项目内路径
                    "original_path": image_path,  # 保存原始路径供参考
                    "filename": os.path.basename(image_path),
                    "hash": file_hash
                }],
                ids=[file_hash]
            )

            return True

        except Exception as e:
            print(f"索引图像失败 {image_path}: {e}")
            return False

    def index_directory(self, directory: str) -> int:
        """批量索引目录中的图像"""
        directory = Path(directory)
        if not directory.exists():
            print(f"目录不存在: {directory}")
            return 0

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        count = 0

        print(f"开始索引目录: {directory}")
        print(f"图像将保存到: {IMAGES_DIR}")

        for img_file in directory.rglob("*"):
            if img_file.suffix.lower() in image_extensions:
                try:
                    if self.index_image(str(img_file)):
                        count += 1
                        if count % 10 == 0:
                            print(f"已索引 {count} 张图像...")
                except Exception as e:
                    print(f"跳过 {img_file}: {e}")

        print(f"索引完成，共处理 {count} 张图像")
        return count

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """以文搜图"""
        # 获取查询文本的CLIP嵌入
        query_embedding = self.model_manager.get_query_embedding(query)

        # 查询向量数据库
        try:
            results = self.vector_db.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["metadatas", "distances"]
            )

            if not results['ids'][0]:
                return []

            # 返回结果（使用项目内的路径）
            return list(zip(
                [metadata['path'] for metadata in results['metadatas'][0]],
                [1.0 - (distance / 2.0) for distance in results['distances'][0]]
            ))
        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    def list_images(self) -> List[str]:
        """列出所有已索引的图像"""
        try:
            results = self.vector_db.get()
            return [metadata['path'] for metadata in results['metadatas']]
        except:
            return []