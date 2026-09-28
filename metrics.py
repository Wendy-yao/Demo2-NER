from typing import Iterable, List, Set, Tuple

# 用 (type, start, end) 三元组表示实体
Entity = Tuple[str, int, int]


class EntityExtractor:
    """从 BIO 标签序列中抽取实体集合，遵循常见 B/I 规则。"""

    @staticmethod
    def extract(tags: List[str]) -> Set[Entity]:
        spans: Set[Entity] = set()
        i, n = 0, len(tags)
        while i < n:
            tag = tags[i]
            if tag == "O" or "-" not in tag:
                i += 1
                continue
            prefix, etype = tag.split("-", 1)
            if prefix == "B":
                start = i
                j = i + 1
                while j < n and tags[j] == f"I-{etype}":
                    j += 1
                spans.add((etype, start, j - 1))
                i = j
            else:
                # 孤立 I-X：按 B-X 处理
                start = i
                j = i + 1
                while j < n and tags[j] == f"I-{etype}":
                    j += 1
                spans.add((etype, start, j - 1))
                i = j
        return spans


class NERMetrics:
    """累计 P/R/F1，按实体级统计。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.n_pred = 0
        self.n_gold = 0

    def update(self, pred_tags: Iterable[List[str]], gold_tags: Iterable[List[str]]):
        for p, g in zip(pred_tags, gold_tags):
            pred_set = EntityExtractor.extract(list(p))
            gold_set = EntityExtractor.extract(list(g))
            self.tp += len(pred_set & gold_set)
            self.n_pred += len(pred_set)
            self.n_gold += len(gold_set)

    def compute(self) -> dict:
        precision = self.tp / self.n_pred if self.n_pred else 0.0
        recall = self.tp / self.n_gold if self.n_gold else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support_pred": self.n_pred,
            "support_gold": self.n_gold,
        }

    def report(self) -> str:
        m = self.compute()
        return (f"precision={m['precision']:.4f} "
                f"recall={m['recall']:.4f} "
                f"f1={m['f1']:.4f}")
