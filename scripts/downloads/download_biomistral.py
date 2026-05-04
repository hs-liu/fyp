from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='BioMistral/BioMistral-7B',
    local_dir='/vol/bitbucket/hl2622/fyp/models/biomistral-7b',
)
print('Done!')