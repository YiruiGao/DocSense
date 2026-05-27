"""日志配置模块"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(
    log_level: str = "INFO",
    log_dir: str | Path = "data/logs",
    enable_file_logging: bool = True
) -> None:
    """
    配置日志系统

    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        log_dir: 日志目录路径
        enable_file_logging: 是否启用文件日志
    """
    # 确定日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 确定日志目录
    log_path = Path(log_dir)

    # 创建日志目录
    if enable_file_logging:
        log_path.mkdir(parents=True, exist_ok=True)

    # 配置根日志
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除现有处理器
    root_logger.handlers.clear()

    # 日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    # 文件处理器 (可选)
    if enable_file_logging:
        file_handler = RotatingFileHandler(
            log_path / 'app.log',
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

    # 设置第三方库日志级别 (降低噪音)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("FlagEmbedding").setLevel(logging.WARNING)



    # 记录日志配置完成
    root_logger.info(f"日志系统初始化完成: 级别={logging.getLevelName(level)}, 文件日志={enable_file_logging}")


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器

    Args:
            name: 日志记录器名称 (通常使用 __name__)

        Returns:
            logging.Logger 实例
    """
    return logging.getLogger(name)


class RequestContextFilter:
    """请求上下文过滤器, 为日志添加请求ID"""

    def filter(self, record: logging.LogRecord) -> bool:
        # 如果有 request_id 属性, 则添加到日志中
        if hasattr(record, 'request_id'):
            return True
        return False


class RequestFormatter(logging.Formatter):
    """请求格式化器, 包含请求ID"""

    def format(self, record: logging.LogRecord) -> str:
        # 基础格式
        base_format = super().format(record)

        # 添加请求ID (如果有)
        if hasattr(record, 'request_id'):
            return f"[{record.request_id}] {base_format}"
        return base_format


def add_request_id(logger: logging.Logger, request_id: str) -> logging.Logger:
    """
    为日志记录器添加请求ID过滤器

    Args:
            logger: 日志记录器
            request_id: 请求ID
    """
    # 创建新的处理器, 包含请求ID
    for handler in logger.handlers:
        # 创建带请求ID的格式化器
        formatter = RequestFormatter(fmt=handler.formatter._fmt)
        handler.setFormatter(formatter)

    return logger


# 初始化时调用 (在应用启动时调用)
# setup_logging()
