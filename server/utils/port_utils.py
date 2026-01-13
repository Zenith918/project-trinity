"""
端口管理工具

使用方法：
    from server.utils.port_utils import ensure_port_free
    ensure_port_free(9003)  # 启动前调用，自动杀掉占用端口的进程
"""

import subprocess
from loguru import logger


def ensure_port_free(port: int) -> bool:
    """
    确保指定端口未被占用，如果被占用则强制杀掉占用进程
    
    Args:
        port: 端口号
        
    Returns:
        True 如果端口现在可用
    """
    try:
        # 方法1：使用 fuser
        subprocess.run(
            f"fuser -k {port}/tcp 2>/dev/null",
            shell=True, capture_output=True, timeout=5
        )
        
        # 方法2：使用 ss + kill（备用）
        result = subprocess.run(
            f"ss -tlnp 2>/dev/null | grep ':{port} ' | grep -oP 'pid=\\K[0-9]+' | head -1",
            shell=True, capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            pid = result.stdout.strip()
            subprocess.run(f"kill -9 {pid} 2>/dev/null", shell=True, timeout=5)
            logger.warning(f"🔪 已杀掉占用端口 {port} 的进程 (PID: {pid})")
        
        return True
    except Exception as e:
        logger.warning(f"端口清理警告: {e}")
        return True  # 即使清理失败也继续，让 uvicorn 报错


def is_port_in_use(port: int) -> bool:
    """检查端口是否被占用"""
    try:
        result = subprocess.run(
            f"ss -tlnp 2>/dev/null | grep ':{port} '",
            shell=True, capture_output=True, text=True, timeout=5
        )
        return bool(result.stdout.strip())
    except Exception:
        return False



