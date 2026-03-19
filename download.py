from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-2B-Instruct",
    local_dir="../download/Qwen3-VL-2B-Instruct",
    max_workers=16,
)