import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DomainClassifier:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        label_map     = json.load(open(f"{model_dir}/label_map.json"))
        self.id2label = {int(k): v for k, v in label_map["id2label"].items()}

        import os
        temp_path = f"{model_dir}/temperature.json"
        self.temperature = json.load(open(temp_path))["temperature"] if os.path.exists(temp_path) else 1.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, query: str):
        enc = self.tokenizer(
            query, max_length=256, truncation=True,
            padding="max_length", return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs  = torch.softmax(logits / self.temperature, dim=-1)

        pred_id    = probs.argmax().item()
        confidence = probs[0][pred_id].item()
        domain     = self.id2label[pred_id]

        return {
            "domain"    : domain,
            "confidence": round(confidence, 4),
            "all_probs" : {
                self.id2label[i]: round(p, 4)
                for i, p in enumerate(probs[0].tolist())
            }
        }
