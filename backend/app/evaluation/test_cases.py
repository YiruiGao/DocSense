"""测试用例定义"""
from typing import List, Optional
from pydantic import BaseModel
from dataclasses import dataclass


from enum import Enum


class QuestionDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class TestCase:
    """测试用例"""
    id: str
    question: str
    expected_chunks: List[str]  # 期望命中的chunk关键词
    expected_page_numbers: List[int]  # 期望的页码
    difficulty: QuestionDifficulty = QuestionDifficulty.EASY
    category: Optional[str] = None  # 分类标签
    document_id: Optional[str] = None  # 关联的文档ID


class TestCaseSet(BaseModel):
    """测试用例集合"""
    id: str
    name: str
    description: Optional[str] = None
    test_cases: List[TestCase]
    created_at: Optional[str] = None


    document_id: Optional[str] = None  # 关联的文档


    @classmethod
    def create_default(cls) -> "TestCaseSet":
        """创建默认测试集"""
        return TestCaseSet(
            id="default",
            name="默认测试集",
            description="用于评估RAG系统的基础测试用例",
            test_cases=[
                # 简单问题
                TestCase(
                    id="q1",
                    question="文档的主要主题是什么？",
                    expected_chunks=["主题", "主要内容", "核心"],
                    expected_page_numbers=[1],
                    difficulty=QuestionDifficulty.EASY,
                    category="basic"
                ),
                # 中等问题
                TestCase(
                    id="q2",
                    question="文档中提到了哪些关键结论？",
                    expected_chunks=["结论", "结果", "发现"],
                    expected_page_numbers=[2, 3],
                    difficulty=QuestionDifficulty.MEDIUM,
                    category="intermediate"
                ),
                # 困难问题
                TestCase(
                    id="q3",
                    question="第5页第三段的具体内容是什么？",
                    expected_chunks=["第5页", "第三段"],
                    expected_page_numbers=[5],
                    difficulty=QuestionDifficulty.HARD,
                    category="detailed"
                ),
                # 页码定位问题
                TestCase(
                    id="q4",
                    question="第3页讲了什么？",
                    expected_chunks=["第3页"],
                    expected_page_numbers=[3],
                    difficulty=QuestionDifficulty.EASY,
                    category="location"
                ),
                # 概念定义问题
                TestCase(
                    id="q5",
                    question="什么是主要特点？",
                    expected_chunks=["特点", "特征", "特性"],
                    expected_page_numbers=[1, 2],
                    difficulty=QuestionDifficulty.MEDIUM,
                    category="concept"
                ),
                # 比较问题
                TestCase(
                    id="q6",
                    question="文档开头和结尾有什么区别？",
                    expected_chunks=["开头", "结尾", "区别"],
                    expected_page_numbers=[1],
                    difficulty=QuestionDifficulty.HARD,
                    category="comparison"
                ),
                # 数量问题
                TestCase(
                    id="q7",
                    question="文档总共有多少页？",
                    expected_chunks=["页数", "总数", "多少"],
                    expected_page_numbers=[1],
                    difficulty=QuestionDifficulty.EASY,
                    category="quantitative"
                ),
                # 结构问题
                TestCase(
                    id="q8",
                    question="文档的结构是怎样的？",
                    expected_chunks=["结构", "组织", "框架"],
                    expected_page_numbers=[1, 2],
                    difficulty=QuestionDifficulty.MEDIUM,
                    category="structural"
                ),
                # 细节问题
                TestCase(
                    id="q9",
                    question="有哪些具体的示例或案例？",
                    expected_chunks=["示例", "案例", "例子"],
                    expected_page_numbers=[3, 4, 5],
                    difficulty=QuestionDifficulty.HARD,
                    category="detailed"
                ),
                # 总结问题
                TestCase(
                    id="q10",
                    question="文档的核心观点是什么？",
                    expected_chunks=["核心", "观点", "主旨"],
                    expected_page_numbers=[1],
                    difficulty=QuestionDifficulty.MEDIUM,
                    category="summary"
                ),
            ]
        )


# 全局默认测试集
DEFAULT_TEST_CASES = TestCaseSet.create_default()
