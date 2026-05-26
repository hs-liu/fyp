from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='BioMistral/BioMistral-7B',
    local_dir='/vol/bitbucket/hl2622/fyp/src/models/biomistral-7b',
)
snapshot_download(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    local_dir='/vol/bitbucket/hl2622/fyp/src/models/llama-3.1-8b-instruct',
)
snapshot_download(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    local_dir='/vol/bitbucket/hl2622/fyp/src/models/qwen2.5-7b-instruct',
)
print('Done!')