import threading
import time
import subprocess
from loguru import logger
from config import settings

class SystemMonitor:
    def __init__(self, interval: int = 2):
        self.interval = interval
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("🛡️ 系统资源监控已启动 (VRAM Monitor)")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _monitor_loop(self):
        while self.running:
            try:
                # 使用 nvidia-smi 查询显存
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    output = result.stdout.strip()
                    used, total = map(int, output.split(','))
                    
                    if used > settings.server.vram_critical_threshold:
                        logger.critical(f"🔥 VRAM CRITICAL: {used}MB / {total}MB (Potential OOM!)")
                    elif used > settings.server.vram_warning_threshold:
                        logger.warning(f"⚠️ VRAM WARNING: {used}MB / {total}MB")
                    
                    # 可以在这里扩展 CPU/RAM 监控
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            time.sleep(self.interval)


