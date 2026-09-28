import torch
from transformers import AutoTokenizer

from module import TaggingNet


class Predictor:
    """加载已训练权重，对单句做字级 NER 标注。"""

    def __init__(self, ckpt_path: str, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.backbone_path = ckpt["backbone_path"]
        self.n_tags = ckpt["n_tags"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_path)

        self.model = TaggingNet(self.backbone_path, self.n_tags).to(self.device).eval()
        self.model.load_state_dict(ckpt["model_state_dict"])

        # 从 ckpt 里读 label 表（如果存了的话）
        self.id2tag = ckpt.get("id2tag", None)

    @torch.no_grad()
    def tag(self, text: str):
        chars = list(text)
        enc = self.tokenizer(chars, is_split_into_words=True, return_tensors="pt").to(self.device)
        logits = self.model(**enc)["logits"]
        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()

        wids = enc.word_ids()
        seen, pairs = set(), []
        for wid, pid in zip(wids, pred_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            label = self.id2tag[pid] if self.id2tag else str(pid)
            pairs.append((chars[wid], label))
        return pairs
