"""
对话日志记录器
按天归档，方便获取测试问答数据
"""

import os
import json
from datetime import datetime
from pathlib import Path
from loguru import logger

LOG_DIR = Path("/workspace/project-trinity/project-trinity/logs/conversations")


class ConversationLogger:
    """对话日志管理器"""
    
    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    def _get_today_file(self) -> Path:
        """获取今天的日志文件"""
        today = datetime.now().strftime("%Y-%m-%d")
        return LOG_DIR / f"{today}.jsonl"
    
    def log(self, user_input: str, ai_response: str, metadata: dict = None):
        """
        记录一次对话
        
        Args:
            user_input: 用户输入
            ai_response: AI 回复
            metadata: 额外元数据 (延迟、tokens 等)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": ai_response,
            "metadata": metadata or {}
        }
        
        log_file = self._get_today_file()
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"对话日志写入失败: {e}")
    
    def get_today_conversations(self) -> list:
        """获取今天的所有对话"""
        return self.get_conversations_by_date(datetime.now().strftime("%Y-%m-%d"))
    
    def get_conversations_by_date(self, date: str) -> list:
        """
        获取指定日期的对话
        
        Args:
            date: 日期字符串 "YYYY-MM-DD"
        """
        log_file = LOG_DIR / f"{date}.jsonl"
        
        if not log_file.exists():
            return []
        
        conversations = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))
        
        return conversations
    
    def list_available_dates(self) -> list:
        """列出所有有日志的日期"""
        dates = []
        for f in LOG_DIR.glob("*.jsonl"):
            dates.append(f.stem)  # 文件名就是日期
        return sorted(dates, reverse=True)


# 全局实例
conversation_logger = ConversationLogger()

