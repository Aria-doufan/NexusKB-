from __future__ import annotations

from modelscope import snapshot_download


def main() -> None:
    local_dir = r"D:\Hugging_Face\models\Qwen3-Reranker-0.6B"
    path = snapshot_download(
        "Qwen/Qwen3-Reranker-0.6B",
        local_dir=local_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
