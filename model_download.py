from huggingface_hub import snapshot_download


if __name__ == "__main__":
    snapshot_download(
        repo_id="google/mt5-base",
        local_dir=r"pretrained_models\nlp\mt5-base",
        local_dir_use_symlinks=False,
        cache_dir=r"pretrained_models\nlp\mt5-base",
        ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
        resume_download=True,
    )
  