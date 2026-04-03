"""LLM 调用模块 - 支持 OpenAI-compatible API"""
import httpx
from typing import Optional, List, Dict, Any
from app.common.config import settings
from app.common.logging import get_logger
import json

logger = get_logger(__name__)


class ChatLLM:
    """OpenAI-compatible 大模型调用"""

    def __init__(self):
        self.provider = settings.llm_provider.strip().lower()
        self.api_key = self._get_api_key()
        self.base_url = self._get_base_url().rstrip("/")
        self.model = self._get_model()
        self._client = httpx.AsyncClient(timeout=settings.llm_timeout_seconds)

    def _get_api_key(self) -> Optional[str]:
        if self.provider == "deepseek":
            return settings.deepseek_api_key
        if self.provider == "zai":
            return settings.zai_api_key
        raise LLMError(f"不支持的 LLM_PROVIDER: {settings.llm_provider}")

    def _get_base_url(self) -> str:
        if self.provider == "deepseek":
            return settings.deepseek_base_url
        return settings.zai_base_url

    def _get_model(self) -> str:
        if self.provider == "deepseek":
            return settings.deepseek_model
        return settings.zai_model

    def _provider_payload(self) -> Dict[str, Any]:
        if self.provider != "deepseek":
            return {}

        payload: Dict[str, Any] = {}
        if settings.deepseek_reasoning_effort:
            payload["reasoning_effort"] = settings.deepseek_reasoning_effort
        if settings.deepseek_thinking_enabled:
            payload["thinking"] = {"type": "enabled"}
        return payload

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        生成回复

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            str: 生成的回复
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        if not self.api_key:
            raise LLMError(f"{self.provider} API Key 未配置")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        payload.update(self._provider_payload())

        try:
            # DEBUG: 打印发送给 LLM 的内容
            logger.debug(f"LLM 请求 - Provider: {self.provider}, Model: {self.model}")
            logger.debug(f"LLM 请求 - Messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")
            logger.debug(f"LLM 请求 - Temperature: {temperature}, MaxTokens: {max_tokens}")

            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # DEBUG: 打印 LLM 返回的内容
            logger.debug(f"LLM 响应 - 内容长度: {len(content)} 字符")
            logger.debug(f"LLM 响应 - 内容: {content[:500]}{'...' if len(content) > 500 else ''}")

            return content

        except httpx.HTTPStatusError as e:
            raise LLMError(f"API请求失败: {e.response.status_code} - {e.response.text}")
        except httpx.TimeoutException:
            raise LLMError(f"LLM调用超时 (超过 {self._client.timeout}s)")
        except Exception as e:
            raise LLMError(f"LLM调用失败: {type(e).__name__}: {str(e) or '未知错误'}")

    async def generate_with_sources(
        self,
        question: str,
        chunks: List[Dict[str, Any]]
    ) -> str:
        """
        基于文档块生成带引用的回答

        Args:
            question: 用户问题
            chunks: 检索到的文档块

        Returns:
            str: 带引用的回答
        """
        # 构建上下文
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page_num = chunk.get("metadata", {}).get("page_number", chunk.get("page_number", "?"))
            content = chunk.get("content", "")
            if len(content) > settings.llm_context_chars_per_chunk:
                content = content[:settings.llm_context_chars_per_chunk] + "..."
            context_parts.append(f"[{i}] 第{page_num}页：{content}")

        context = "\n".join(context_parts)

        system_prompt = """你是一个准确的问答助手。基于提供的文档片段回答问题。

严格要求：
1. 只使用提供的文档内容回答
2. 必须在回答中标注引用来源，格式为 [1] [2] 等
3. 如果文档中没有相关信息，明确说"文档中没有相关信息"
4. 不要编造或推断任何内容
5. 回答要简洁准确"""

        user_prompt = f"""文档片段：
{context}

问题：{question}

请直接回答问题，在相关内容处标注引用编号如 [1][2]。"""

        return await self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=settings.llm_max_tokens
        )

    async def rewrite_query(self, query: str) -> str:
        """
        改写查询以更适合检索

        Args:
            query: 原始查询

        Returns:
            str: 改写后的查询
        """
        system_prompt = """你是一个查询优化助手。将用户的问题改写成更适合文档检索的查询语句。
要求：
1. 保持原意
2. 使用更精确的关键词
3. 只返回改写后的查询，不要其他内容"""

        return await self.generate(
            prompt=f"原问题：{query}\n\n改写后：",
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=200
        )

    async def close(self):
        """关闭客户端"""
        await self._client.aclose()


class LLMError(Exception):
    """LLM调用异常"""
    pass


class ZaiLLM(ChatLLM):
    """兼容旧导入名称的 LLM 客户端"""


# 全局单例
llm = ChatLLM()
zai_llm = llm
