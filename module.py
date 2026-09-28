import torch
import torch.nn as nn
from transformers import AutoModel


class TaggingNet(nn.Module):
    """BERT + Dropout + 线性分类头，token-level NER。"""

    def __init__(self, backbone_path: str, n_tags: int, settings=None):
        super().__init__()
        self.n_tags = n_tags

        self.bert = AutoModel.from_pretrained(backbone_path)
        hidden = self.bert.config.hidden_size

        dropout = getattr(settings, "dropout", 0.1) if settings else 0.1
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_tags)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        seq = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        seq = self.dropout(seq)
        logits = self.head(seq)

        out = {"logits": logits, "predictions": torch.argmax(logits, dim=-1)}
        if labels is not None:
            out["loss"] = self.loss_fn(logits.view(-1, self.n_tags), labels.view(-1))
        return out

    @torch.no_grad()
    def predict(self, input_ids, attention_mask):
        self.eval()
        return self.forward(input_ids, attention_mask)["predictions"]

    def freeze_bert(self, freeze=True):
        for p in self.bert.parameters():
            p.requires_grad = not freeze

    def freeze_embeddings(self, freeze=True):
        for p in self.bert.embeddings.parameters():
            p.requires_grad = not freeze
