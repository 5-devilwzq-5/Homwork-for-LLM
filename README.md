# Homwork-for-LLM
BJTU 2025 多模态课程大作业 构建一个本地多模态 AI 智能助手

## 1 核心功能要求：

### 1.1 智能文献管理
*   **语义搜索**: 支持使用自然语言提问（如“Transformer 的核心架构是什么？”）。系统需基于语义理解返回最相关的论文文件，进阶要求可返回具体的论文片段或页码。
   
*   **自动分类与整理**:
    *   **单文件处理**: 添加新论文时，根据指定的主题（如 "CV, NLP, RL"）自动分析内容，将其归类并移动到对应的子文件夹中。
        <img width="1376" height="157" alt="d7a49462e791374919357df1dd73e093" src="https://github.com/user-attachments/assets/8b7a0cf1-b861-4ca8-845b-42ea9a273231" />
        <img width="1726" height="140" alt="ac69ea75189d1f1819caf18c45df9260" src="https://github.com/user-attachments/assets/7e8fd108-d1c2-45aa-9ca3-3f6f5c00551b" />
    *   **批量整理**: 支持对现有的混乱文件夹进行“一键整理”，自动扫描所有 PDF，识别主题并归档到相应目录。
        <img width="982" height="510" alt="c668d158b409fabe6685a8db72aea409" src="https://github.com/user-attachments/assets/5890b0ec-8f51-4af7-8c42-863e3ad06f8b" />

*   **文件索引**: 支持仅返回相关文件列表，方便快速定位所需文献。
<img width="1305" height="578" alt="3009f6c44237cb63f847c8c1c9926cea" src="https://github.com/user-attachments/assets/9d3eb67a-9a0f-4ebd-af3a-da73e89dcea9" />

### 1.2 智能图像管理
*   **以文搜图**: 利用多模态图文匹配技术，支持通过自然语言描述（如“海边的日落”）来查找本地图片库中最匹配的图像。
    <img width="816" height="246" alt="434a23f9f9478dc482df7deb186e2202" src="https://github.com/user-attachments/assets/c56afd94-3fc2-4cd3-a921-f77f17e64653" />
    <img width="1856" height="212" alt="78f9fee0b326008edf6c4ed96f11c55f" src="https://github.com/user-attachments/assets/a0e1d04d-c8ed-4d4e-b627-59b32153fa43" />

