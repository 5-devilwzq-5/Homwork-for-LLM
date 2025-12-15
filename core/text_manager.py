"""
文本（论文）管理模块
"""
import os
import fitz  # PyMuPDF
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.utils import embedding_functions

from config import MODEL_CONFIG, VECTOR_DB_CONFIG, PAPER_CATEGORIES, PAPERS_DIR
from utils.pdf_processor import extract_text_from_pdf, split_text
from core.models import ModelManager


class TextManager:
    """文本管理器"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.vector_db = self._init_vector_db()

    def _init_vector_db(self):
        """初始化向量数据库"""
        chroma_client = chromadb.PersistentClient(
            path=VECTOR_DB_CONFIG["persist_directory"]
        )

        # 创建或获取集合
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_CONFIG["text_embedding"]["name"]
        )

        try:
            collection = chroma_client.get_collection(
                name=VECTOR_DB_CONFIG["text_collection"],
                embedding_function=embedding_func
            )
        except:
            collection = chroma_client.create_collection(
                name=VECTOR_DB_CONFIG["text_collection"],
                embedding_function=embedding_func
            )

        return collection

    def add_paper(self, pdf_path: str, topics: str = None) -> Dict[str, Any]:
        """添加单篇论文"""
        # 提取文本
        text = extract_text_from_pdf(pdf_path)
        if not text or len(text) < 100:
            raise ValueError("无法从PDF中提取有效文本")

        # 提取标题（使用第一行作为标题）
        title = text[:100].split('\n')[0].strip()[:50]

        # 分类
        if topics:
            categories = [t.strip() for t in topics.split(",")]
        else:
            categories = PAPER_CATEGORIES

        category = self.model_manager.classify_paper(text, categories)

        # 生成唯一ID
        file_hash = hashlib.md5(f"{pdf_path}{os.path.getmtime(pdf_path)}".encode()).hexdigest()

        # 将文本分块
        chunks = split_text(text)

        # 为每个块生成嵌入并存储
        chunk_ids = []
        chunk_texts = []
        metadatas = []

        for i, chunk in enumerate(chunks[:20]):  # 限制前20个块
            chunk_id = f"{file_hash}_{i}"
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk)

            metadatas.append({
                "paper_id": file_hash,
                "title": title,
                "category": category,
                "path": pdf_path,
                "chunk_index": i
            })

        # 添加到向量数据库
        self.vector_db.add(
            documents=chunk_texts,
            metadatas=metadatas,
            ids=chunk_ids
        )

        # 复制到分类目录
        target_dir = PAPERS_DIR / category
        target_dir.mkdir(exist_ok=True)

        import shutil
        target_path = target_dir / Path(pdf_path).name
        shutil.copy2(pdf_path, target_path)

        return {
            "id": file_hash,
            "title": title,
            "category": category,
            "path": str(target_path),
            "chunks": len(chunks)
        }

    def process_paper_directory(self, directory: str, topics: str = None) -> List[Dict]:
        """批量处理论文目录"""
        results = []
        directory = Path(directory)

        for pdf_file in directory.glob("**/*.pdf"):
            try:
                print(f"处理: {pdf_file.name}")
                result = self.add_paper(str(pdf_file), topics)
                results.append(result)
            except Exception as e:
                print(f"处理失败 {pdf_file}: {e}")

        return results

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """搜索论文"""
        try:
            # 查询向量数据库
            results = self.vector_db.query(
                query_texts=[query],
                n_results=top_k * 3,
                include=["documents", "metadatas", "distances"]
            )

            if not results['ids'][0]:
                print("没有找到相关论文")
                return []

            # 按论文去重，保留最相关的结果
            papers_dict = {}

            for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
            ):
                paper_id = metadata['paper_id']
                score = 1.0 - (distance / 2.0)  # 转换距离为相似度

                if paper_id not in papers_dict or score > papers_dict[paper_id]['score']:
                    # 确保有 snippet 字段
                    snippet = doc[:500] if doc else "无可用片段"
                    papers_dict[paper_id] = {
                        **metadata,
                        'snippet': snippet,
                        'score': score
                    }

            # 按相似度排序
            sorted_papers = sorted(
                papers_dict.values(),
                key=lambda x: x['score'],
                reverse=True
            )[:top_k]

            return [(paper, paper['score']) for paper in sorted_papers]

        except Exception as e:
            print(f"搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def organize_directory(self, directory: str) -> int:
        """整理目录中的论文"""
        return len(self.process_paper_directory(directory))

    def list_papers(self) -> List[Dict]:
        """列出所有已索引的论文"""
        # 获取所有唯一的论文
        all_results = self.vector_db.get()

        papers = {}
        for metadata in all_results['metadatas']:
            paper_id = metadata['paper_id']
            if paper_id not in papers:
                papers[paper_id] = {
                    'title': metadata['title'],
                    'category': metadata['category'],
                    'path': metadata['path']
                }

        return list(papers.values())