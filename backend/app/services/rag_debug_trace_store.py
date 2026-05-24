import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from app.schemas.rag_debug import RagDebugTrace


DEFAULT_TRACE_DIR = Path(__file__).resolve().parents[2] / "data" / "rag_debug_traces"


class DebugTraceStore:
    def __init__(self, base_dir: str | Path = DEFAULT_TRACE_DIR):
        self.base_dir = Path(base_dir)

    async def save(self, trace: RagDebugTrace) -> None:
        await asyncio.to_thread(self._save_sync, trace)

    async def get(self, debug_id: str) -> RagDebugTrace | None:
        return await asyncio.to_thread(self._get_sync, debug_id)

    def _save_sync(self, trace: RagDebugTrace) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.base_dir / f"{datetime.now(UTC).date().isoformat()}.jsonl"
        payload = trace.model_dump_json()
        with file_path.open("a", encoding="utf-8") as file:
            file.write(payload + "\n")

    def _get_sync(self, debug_id: str) -> RagDebugTrace | None:
        if not self.base_dir.exists():
            return None
        for file_path in sorted(self.base_dir.glob("*.jsonl"), reverse=True):
            with file_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("debug_id") == debug_id:
                        return RagDebugTrace.model_validate(data)
        return None


debug_trace_store = DebugTraceStore()
