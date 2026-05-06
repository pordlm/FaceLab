"""
FaceLab profiling 工具模块。
由 FACELAB_PROFILE=1 环境变量控制，默认关闭。
不引入任何外部依赖。
"""
import json
import os
import time
import uuid
from pathlib import Path
from contextlib import contextmanager


_PROFILE_ENABLED: bool | None = None
_PROFILE_LOG: "ProfileLog | None" = None


def is_profiling_enabled() -> bool:
    """检查 FACELAB_PROFILE 环境变量是否设置为 '1'"""
    global _PROFILE_ENABLED
    if _PROFILE_ENABLED is None:
        _PROFILE_ENABLED = os.getenv("FACELAB_PROFILE") == "1"
    return _PROFILE_ENABLED


class ProfileLog:
    """线程安全的 JSONL profiling 日志写入器"""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "a", encoding="utf-8")
        self.run_id = uuid.uuid4().hex[:8]
        self.pid = os.getpid()

    def _event_base(self, event: str, **fields) -> dict:
        return {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int((time.time() % 1) * 1e6):06d}",
            "run_id": self.run_id,
            "pid": self.pid,
            "event": event,
            **fields,
        }

    def log(self, event: str, **fields) -> None:
        """写一行 JSONL profiling 日志"""
        if not is_profiling_enabled():
            return
        record = self._event_base(event, **fields)
        self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


def get_profile_log() -> "ProfileLog | None":
    """获取全局 profiling 日志实例"""
    global _PROFILE_LOG
    if not is_profiling_enabled():
        return None
    if _PROFILE_LOG is None:
        from config import BASE_DIR
        output_dir = BASE_DIR / "profile_logs"
        _PROFILE_LOG = ProfileLog(output_dir / "profile.jsonl")
    return _PROFILE_LOG


@contextmanager
def timed(log: "ProfileLog | None", event: str, **extra):
    """
    上下文管理器：自动记录 start/end 并计算 elapsed_ms。
    如果 log 为 None，计时代码不执行。

    用法:
        with timed(log, "my_event", module="foo", function="bar"):
            do_work()
    """
    if log is None:
        yield
        return

    log.log(event, phase="start", **extra)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log.log(event, phase="end", elapsed_ms=round(elapsed_ms, 3), **extra)


def log_measure(log: "ProfileLog | None", event: str, elapsed_ms: float, **extra):
    """记录单次测量事件（不带 start/end，直接记录耗时）"""
    if log is None:
        return
    log.log(event, phase="measure", elapsed_ms=round(elapsed_ms, 3), **extra)
