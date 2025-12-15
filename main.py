import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional
from core.text_manager import TextManager
from core.image_manager import ImageManager
from core.models import ModelManager
from utils.file_utils import setup_data_directories


def setup_argparse():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(
        description="本地多模态AI助手 - 文献与图像智能管理",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 添加/分类论文
    paper_parser = subparsers.add_parser('add_paper', help='添加并分类论文')
    paper_parser.add_argument('path', help='论文文件或目录路径')
    paper_parser.add_argument('--topics', help='自定义主题（逗号分隔），如 "CV,NLP,RL"')
    paper_parser.add_argument('--batch', action='store_true', help='批量处理目录')

    # 搜索论文
    search_parser = subparsers.add_parser('search_paper', help='搜索论文')
    search_parser.add_argument('query', help='搜索查询语句')
    search_parser.add_argument('--top_k', type=int, default=3, help='返回结果数量')
    search_parser.add_argument('--show-snippets', action='store_true', help='显示相关片段')

    # 整理论文库
    organize_parser = subparsers.add_parser('organize_papers', help='整理论文库')
    organize_parser.add_argument('directory', help='要整理的目录路径')

    # 添加图像
    image_parser = subparsers.add_parser('add_image', help='添加图像到索引')
    image_parser.add_argument('path', help='图像文件或目录路径')
    image_parser.add_argument('--batch', action='store_true', help='批量处理目录')

    # 搜索图像
    img_search_parser = subparsers.add_parser('search_image', help='以文搜图')
    img_search_parser.add_argument('query', help='图像描述')
    img_search_parser.add_argument('--top_k', type=int, default=3, help='返回结果数量')

    # 设置API
    api_parser = subparsers.add_parser('set_api', help='设置API密钥')
    api_parser.add_argument('--gemini', help='Gemini API密钥')
    api_parser.add_argument('--openai', help='OpenAI API密钥')

    # 列出索引
    list_parser = subparsers.add_parser('list', help='列出索引内容')
    list_parser.add_argument('type', choices=['papers', 'images'], help='类型')

    return parser


def main():
    """主函数"""
    # 初始化
    parser = setup_argparse()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 设置数据目录
    setup_data_directories()

    # 初始化模型管理器
    print("正在初始化模型...")
    model_manager = ModelManager()

    try:
        if args.command == 'add_paper':
            # 添加论文
            text_manager = TextManager(model_manager)
            if args.batch or os.path.isdir(args.path):
                results = text_manager.process_paper_directory(args.path, args.topics)
                print(f"成功处理 {len(results)} 篇论文")
            else:
                result = text_manager.add_paper(args.path, args.topics)
                print(f"论文添加成功: {result['title']}")
                print(f"分类到: {result['category']}")

        elif args.command == 'search_paper':
            # 搜索论文
            text_manager = TextManager(model_manager)
            results = text_manager.search(args.query, top_k=args.top_k)

            print(f"\n找到前 {len(results)} 篇相关论文:")
            for i, (paper, score) in enumerate(results, 1):
                print(f"\n{i}. {paper['title']} (相似度: {score:.3f})")
                print(f"   路径: {paper['path']}")
                print(f"   分类: {paper['category']}")
                if args.show_snippets and 'snippet' in paper:
                    print(f"   相关片段: {paper['snippet'][:500]}...")
                elif args.show_snippets and 'snippets' in paper:
                    # 兼容旧版本
                    print(f"   相关片段: {paper['snippets'][0][:300]}...")

        elif args.command == 'organize_papers':
            # 整理论文库
            text_manager = TextManager(model_manager)
            count = text_manager.organize_directory(args.directory)
            print(f"整理完成，处理了 {count} 篇论文")

        elif args.command == 'add_image':
            # 添加图像
            image_manager = ImageManager(model_manager)
            if args.batch or os.path.isdir(args.path):
                count = image_manager.index_directory(args.path)
                print(f"成功索引 {count} 张图像")
            else:
                image_manager.index_image(args.path)
                print(f"图像索引成功: {args.path}")

        elif args.command == 'search_image':
            # 搜索图像
            image_manager = ImageManager(model_manager)
            results = image_manager.search(args.query, top_k=args.top_k)

            print(f"\n找到前 {len(results)} 张相关图像:")
            for i, (img_path, score) in enumerate(results, 1):
                print(f"{i}. {img_path} (相似度: {score:.3f})")

        elif args.command == 'set_api':
            # 设置API密钥
            from config import save_api_keys
            save_api_keys(gemini=args.gemini, openai=args.openai)
            print("API密钥已保存")

        elif args.command == 'list':
            # 列出索引内容
            if args.type == 'papers':
                text_manager = TextManager(model_manager)
                papers = text_manager.list_papers()
                print(f"已索引 {len(papers)} 篇论文:")
                for paper in papers:
                    print(f"  - {paper['title']} ({paper['category']})")
            else:
                image_manager = ImageManager(model_manager)
                images = image_manager.list_images()
                print(f"已索引 {len(images)} 张图像")
                for img in images[:10]:  # 只显示前10个
                    print(f"  - {img}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()