"""
文件处理工具
"""
import os
import shutil
from pathlib import Path
from config import DATA_DIR, PAPERS_DIR, IMAGES_DIR, VECTOR_DB_DIR


def setup_data_directories():
    """设置数据目录结构"""
    directories = [DATA_DIR, PAPERS_DIR, IMAGES_DIR, VECTOR_DB_DIR]

    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)

    # 为每个论文分类创建子目录
    from config import PAPER_CATEGORIES
    for category in PAPER_CATEGORIES:
        category_dir = PAPERS_DIR / category.replace(" ", "_")
        category_dir.mkdir(exist_ok=True)

    return directories


def safe_copy(src: str, dst_dir: Path) -> str:
    """安全复制文件，避免文件名冲突"""
    src_path = Path(src)
    dst_path = dst_dir / src_path.name

    # 如果文件已存在，添加数字后缀
    counter = 1
    while dst_path.exists():
        stem = src_path.stem
        suffix = src_path.suffix
        dst_path = dst_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    shutil.copy2(src, dst_path)
    return str(dst_path)


def get_file_info(filepath: str) -> dict:
    """获取文件信息"""
    path = Path(filepath)
    stat = path.stat()

    return {
        "name": path.name,
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "extension": path.suffix.lower()
    }