"""语义分块模块 - 基于 Token 的智能分块"""
import tiktoken
from typing import List, Tuple, Optional
from dataclasses import dataclass
import re
from app.common.config import settings


@dataclass
class ChunkResult:
    """分块结果"""
    content: str
    token_count: int
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int


class SemanticChunker:
    """语义分块器"""

    def __init__(
        self,
        min_tokens: int = None,
        max_tokens: int = None,
        overlap: int = None,
        encoding_name: str = "cl100k_base"  # GPT-4/ChatGPT 使用的编码
    ):
        self.min_tokens = min_tokens or settings.chunk_min_tokens
        self.max_tokens = max_tokens or settings.chunk_max_tokens
        self.overlap = overlap or settings.chunk_overlap

        # 初始化 tokenizer
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            # 降级到简单的字符计数
            self.encoding = None

        # 句子分隔符（支持中英文）
        self._sentence_endings = re.compile(r'[。！？.!?]\s*')

    def chunk_text(
        self,
        text: str,
        page_number: int,
        start_chunk_index: int = 0
    ) -> List[ChunkResult]:
        """
        对单个页面的文本进行分块

        Args:
            text: 要分块的文本
            page_number: 页码
            start_chunk_index: 起始分块索引

        Returns:
            List[ChunkResult]: 分块结果列表
        """
        if not text.strip():
            return []

        # 按段落分割
        paragraphs = self._split_paragraphs(text)

        chunks = []
        current_chunk_index = start_chunk_index
        current_chunk_text = ""
        current_tokens = 0
        start_char = 0
        char_position = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            # 如果单个段落就超过最大token，需要进一步分割
            if para_tokens > self.max_tokens:
                # 先保存当前累积的内容
                if current_chunk_text:
                    chunks.append(ChunkResult(
                        content=current_chunk_text.strip(),
                        token_count=current_tokens,
                        page_number=page_number,
                        chunk_index=current_chunk_index,
                        start_char=start_char,
                        end_char=char_position
                    ))
                    current_chunk_index += 1
                    current_chunk_text = ""
                    current_tokens = 0

                # 分割大段落
                sub_chunks = self._split_large_paragraph(para, page_number, current_chunk_index)
                chunks.extend(sub_chunks)
                current_chunk_index += len(sub_chunks)
                # 重置当前累积的内容
                current_chunk_text = ""
                current_tokens = 0

            # 如果添加这个段落会超过最大token
            elif current_tokens + para_tokens > self.max_tokens:
                # 检查当前累积是否满足最小token要求
                if current_tokens >= self.min_tokens or not current_chunk_text:
                    # 保存当前chunk
                    if current_chunk_text:
                        chunks.append(ChunkResult(
                            content=current_chunk_text.strip(),
                            token_count=current_tokens,
                            page_number=page_number,
                            chunk_index=current_chunk_index,
                            start_char=start_char,
                            end_char=char_position
                        ))
                        current_chunk_index += 1

                    # 开始新的chunk，添加overlap
                    if self.overlap > 0 and current_chunk_text:
                        overlap_text = self._get_overlap_text(current_chunk_text)
                        current_chunk_text = overlap_text + " " + para
                        current_tokens = self._count_tokens(current_chunk_text)
                        start_char = char_position - len(overlap_text)
                    else:
                        current_chunk_text = para
                        current_tokens = para_tokens
                        start_char = char_position
                else:
                    # 当前内容太少，继续添加
                    current_chunk_text += " " + para
                    current_tokens += para_tokens

            else:
                # 添加到当前chunk
                if current_chunk_text:
                    current_chunk_text += " " + para
                else:
                    current_chunk_text = para
                    start_char = char_position
                current_tokens += para_tokens

            char_position += len(para) + 1  # +1 for the space/newline

        # 保存最后一个chunk
        if current_chunk_text.strip():
            chunks.append(ChunkResult(
                content=current_chunk_text.strip(),
                token_count=current_tokens,
                page_number=page_number,
                chunk_index=current_chunk_index,
                start_char=start_char,
                end_char=char_position
            ))

        return chunks

    def chunk_pages(
        self,
        pages: List[Tuple[int, str]]
    ) -> List[ChunkResult]:
        """
        对多个页面进行分块

        Args:
            pages: [(页码, 文本), ...] 列表

        Returns:
            List[ChunkResult]: 所有页面的分块结果
        """
        all_chunks = []
        global_chunk_index = 0

        for page_number, text in pages:
            if not text.strip():
                continue

            page_chunks = self.chunk_text(text, page_number, global_chunk_index)
            all_chunks.extend(page_chunks)
            global_chunk_index += len(page_chunks)

        return all_chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本"""
        # 按换行符分割
        paragraphs = text.split('\n')
        # 过滤空段落并清理
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_large_paragraph(
        self,
        text: str,
        page_number: int,
        start_index: int
    ) -> List[ChunkResult]:
        """分割过大的段落"""
        # 按句子分割
        sentences = self._split_sentences(text)
        chunks = []
        current_text = ""
        current_tokens = 0
        chunk_index = start_index
        start_char = 0
        char_pos = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > self.max_tokens:
                if current_text:
                    chunks.append(ChunkResult(
                        content=current_text.strip(),
                        token_count=current_tokens,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=char_pos
                    ))
                    chunk_index += 1

                current_text = sentence
                current_tokens = sentence_tokens
                start_char = char_pos
            else:
                if current_text:
                    current_text += " " + sentence
                else:
                    current_text = sentence
                    start_char = char_pos
                current_tokens += sentence_tokens

            char_pos += len(sentence) + 1

        if current_text.strip():
            chunks.append(ChunkResult(
                content=current_text.strip(),
                token_count=current_tokens,
                page_number=page_number,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=char_pos
            ))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """按句子分割文本（支持中英文）"""
        # 使用正则分割句子
        parts = self._sentence_endings.split(text)
        # 重新添加句子结束符
        sentences = []
        matches = self._sentence_endings.findall(text)
        for i, part in enumerate(parts[:-1]):
            if i < len(matches):
                sentences.append(part + matches[i])
        if parts[-1].strip():
            sentences.append(parts[-1])
        return [s.strip() for s in sentences if s.strip()]

    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # 降级：按字符估算（中文约1.5字符/token，英文约4字符/token）
            # 简化处理：平均3字符/token
            return max(1, len(text) // 3)

    def _get_overlap_text(self, text: str) -> str:
        """获取重叠部分的文本"""
        if not text:
            return ""

        # 计算overlap对应的token数
        overlap_tokens = min(self.overlap, self._count_tokens(text) // 2)

        if overlap_tokens <= 0:
            return ""

        if self.encoding:
            token_ids = self.encoding.encode(text)
            return self.encoding.decode(token_ids[-overlap_tokens:])

        # Fallback for environments without tiktoken: keep a bounded tail by chars.
        estimated_chars = max(1, overlap_tokens * 3)
        return text[-estimated_chars:]
