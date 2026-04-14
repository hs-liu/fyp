# Corpus

The textbook corpus used in this project is sourced from MedRAG:
https://huggingface.co/datasets/MedRAG/textbooks

## Download

```python
from datasets import load_dataset
import json
from pathlib import Path

dataset = load_dataset("MedRAG/textbooks", split="train")

# Save as jsonl files matching original structure
Path("corpus/textbooks/chunk").mkdir(parents=True, exist_ok=True)
# ... 
```

Or clone directly:
```bash
git clone https://huggingface.co/datasets/MedRAG/textbooks corpus/textbooks
```


## Citation 
@article{xiong2024benchmarking,
    title={Benchmarking Retrieval-Augmented Generation for Medicine}, 
    author={Guangzhi Xiong and Qiao Jin and Zhiyong Lu and Aidong Zhang},
    journal={arXiv preprint arXiv:2402.13178},
    year={2024}
}
