"""PDF 处理模块 - 提取文本并保留页码"""
import fitz  # PyMuPDF
from typing import List, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class PDFPage:
    """PDF 页面数据"""
    page_number: int
    text: str
    char_count: int


@dataclass
class PDFExtractionResult:
    """PDF 提取结果"""
    pages: List[PDFPage]
    total_pages: int
    total_chars: int
    file_name: str


class PDFProcessor:
    """PDF 处理器"""

    def __init__(self, use_ocr: bool = False, ocr_lang: str = "chi_sim+eng"):
        """
        初始化 PDF 处理器

        Args:
            use_ocr: 是否启用 OCR（用于扫描版 PDF）
            ocr_lang: OCR 语言，中文用 chi_sim，英文用 eng，中英文用 chi_sim+eng
        """
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        # 清理文本的正则模式
        self._whitespace_pattern = re.compile(r'\s+')
        self._header_footer_pattern = re.compile(r'页码[:：]?\s*\d+')

    def extract_text_with_pages(self, file_path: str, file_name: str = "") -> PDFExtractionResult:
        """
        从PDF提取文本，保留页码信息

        Args:
            file_path: PDF文件路径
            file_name: 原始文件名

        Returns:
            PDFExtractionResult: 包含所有页面文本的结果
        """
        pages = []
        total_chars = 0

        try:
            doc = fitz.open(file_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")

                # 清理文本
                text = self._clean_text(text)

                if text.strip():  # 只保留有内容的页面
                    page_data = PDFPage(
                        page_number=page_num + 1,  # 页码从1开始
                        text=text,
                        char_count=len(text)
                    )
                    pages.append(page_data)
                    total_chars += len(text)

            doc.close()

        except Exception as e:
            raise PDFProcessingError(f"PDF处理失败: {str(e)}")

        return PDFExtractionResult(
            pages=pages,
            total_pages=len(pages),
            total_chars=total_chars,
            file_name=file_name
        )

    def extract_from_bytes(self, file_bytes: bytes, file_name: str = "") -> PDFExtractionResult:
        """
        从字节数据提取PDF文本

        Args:
            file_bytes: PDF文件的字节数据
            file_name: 原始文件名

        Returns:
            PDFExtractionResult: 包含所有页面文本的结果
        """
        pages = []
        total_chars = 0

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")

                # 如果没有提取到文本且启用了 OCR，则使用 OCR
                if not text.strip() and self.use_ocr:
                    text = self._ocr_page(page)

                # 清理文本
                text = self._clean_text(text)

                if text.strip():
                    page_data = PDFPage(
                        page_number=page_num + 1,
                        text=text,
                        char_count=len(text)
                    )
                    pages.append(page_data)
                    total_chars += len(text)

            doc.close()

        except Exception as e:
            raise PDFProcessingError(f"PDF处理失败: {str(e)}")

        return PDFExtractionResult(
            pages=pages,
            total_pages=len(pages),
            total_chars=total_chars,
            file_name=file_name
        )

    def _ocr_page(self, page) -> str:
        """
        对 PDF 页面进行 OCR

        Args:
            page: fitz Page 对象

        Returns:
            提取的文本
        """
        try:
            import pytesseract
            from pdf2image import convert_from_bytes

            # 将页面渲染为图片
            mat = fitz.Matrix(2.0, 2.0)  # 2x 缩放提高 OCR 精度
            pix = page.get_pixmap(matrix=mat)

            # 转换为 PIL Image
            from PIL import Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # OCR
            text = pytesseract.image_to_string(img, lang=self.ocr_lang)
            return text

        except ImportError:
            raise PDFProcessingError(
                "OCR 依赖未安装。请运行: pip install pytesseract pdf2image && brew install tesseract tesseract-lang"
            )
        except Exception as e:
            raise PDFProcessingError(f"OCR 失败: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """清理提取的文本"""
        # 合并多个空白字符为单个空格
        text = self._whitespace_pattern.sub(' ', text)
        # 移除常见的页眉页脚模式
        text = self._header_footer_pattern.sub('', text)
        # 移除首尾空白
        text = text.strip()
        return text

    def get_page_texts(self, result: PDFExtractionResult) -> List[Tuple[int, str]]:
        """
        获取页面文本列表

        Args:
            result: PDF提取结果

        Returns:
            List[Tuple[int, str]]: (页码, 文本) 列表
        """
        return [(page.page_number, page.text) for page in result.pages]


class PDFProcessingError(Exception):
    """PDF处理异常"""
    pass
