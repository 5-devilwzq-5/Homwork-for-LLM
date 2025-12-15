"""
PDF文本提取工具
"""
import fitz  # PyMuPDF
import re
from typing import List, Optional


def extract_text_from_pdf(pdf_path: str) -> str:
    """从PDF提取文本"""
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"提取PDF文本失败 {pdf_path}: {e}")
        return ""


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """将长文本分割为块"""
    if len(text) <= chunk_size:
        return [text]

    # 按句子分割（简单实现）
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def extract_metadata_from_pdf(pdf_path: str) -> dict:
    """提取PDF元数据"""
    try:
        with fitz.open(pdf_path) as doc:
            metadata = doc.metadata
            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "pages": len(doc)
            }
    except:
        return {}