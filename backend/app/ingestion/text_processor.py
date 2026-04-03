"""文本文件处理模块 - 处理 txt 和 md 文件"""
from typing import List, Tuple
from dataclasses import dataclass
import re


@dataclass
class TextPage:
    """文本页面数据（模拟分页）"""
    page_number: int
    text: str
    char_count: int


@dataclass
class TextExtractionResult:
    """文本提取结果"""
    pages: List[TextPage]
    total_pages: int
    total_chars: int
    file_name: str


class TextProcessor:
    """文本文件处理器"""

    def __init__(self, chars_per_page: int = 3000):
        """
        初始化文本处理器

        Args:
            chars_per_page: 每页字符数（用于模拟分页）
        """
        self.chars_per_page = chars_per_page
        self._whitespace_pattern = re.compile(r'\s+')

    def extract_from_bytes(self, file_bytes: bytes, file_name: str = "") -> TextExtractionResult:
        """
        从字节数据提取文本

        Args:
            file_bytes: 文件的字节数据
            file_name: 原始文件名

        Returns:
            TextExtractionResult: 包含所有页面文本的结果
        """
        # 尝试多种编码
        text = None
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
            try:
                text = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise TextProcessingError("无法识别文件编码")

        # 清理文本
        text = self._clean_text(text)

        # 按段落分割并模拟分页
        pages = self._split_into_pages(text)
        total_chars = sum(p.char_count for p in pages)

        return TextExtractionResult(
            pages=pages,
            total_pages=len(pages),
            total_chars=total_chars,
            file_name=file_name
        )

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 合并多个连续换行为两个换行（保留段落）
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除行尾空白
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        return text.strip()

    def _split_into_pages(self, text: str) -> List[TextPage]:
        """
        将文本分割成虚拟页面

        Args:
            text: 完整文本

        Returns:
            List[TextPage]: 页面列表
        """
        # 按段落分割
        paragraphs = text.split('\n\n')
        pages = []
        current_page_text = ""
        page_number = 1

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前段落加入后不超过每页限制，则加入
            if len(current_page_text) + len(para) + 2 <= self.chars_per_page:
                current_page_text += ("\n\n" if current_page_text else "") + para
            else:
                # 保存当前页
                if current_page_text:
                    pages.append(TextPage(
                        page_number=page_number,
                        text=current_page_text,
                        char_count=len(current_page_text)
                    ))
                    page_number += 1

                # 如果单个段落超过每页限制，需要强制分割
                if len(para) > self.chars_per_page:
                    chunks = self._split_long_paragraph(para)
                    for chunk in chunks:
                        pages.append(TextPage(
                            page_number=page_number,
                            text=chunk,
                            char_count=len(chunk)
                        ))
                        page_number += 1
                    current_page_text = ""
                else:
                    current_page_text = para

        # 保存最后一页
        if current_page_text:
            pages.append(TextPage(
                page_number=page_number,
                text=current_page_text,
                char_count=len(current_page_text)
            ))

        return pages

    def _split_long_paragraph(self, text: str) -> List[str]:
        """分割过长的段落"""
        chunks = []
        while len(text) > self.chars_per_page:
            # 尝试在句子边界分割
            split_pos = text.rfind('。', 0, self.chars_per_page)
            if split_pos == -1:
                split_pos = text.rfind('.', 0, self.chars_per_page)
            if split_pos == -1:
                split_pos = self.chars_per_page
            else:
                split_pos += 1  # 包含句号

            chunks.append(text[:split_pos].strip())
            text = text[split_pos:].strip()

        if text:
            chunks.append(text)

        return chunks

    def get_page_texts(self, result: TextExtractionResult) -> List[Tuple[int, str]]:
        """
        获取页面文本列表

        Args:
            result: 文本提取结果

        Returns:
            List[Tuple[int, str]]: (页码, 文本) 列表
        """
        return [(page.page_number, page.text) for page in result.pages]


class TextProcessingError(Exception):
    """文本处理异常"""
    pass
