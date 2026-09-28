import argparse
import json
import os


class Settings:
    """训练配置：从 JSON 加载，CLI 可覆盖 backbone / dataset / swanlab。"""

    def __init__(self, raw: dict):
        self.backbones: dict = raw["backbones"]
        self.datasets: dict = raw["datasets"]

        t = raw["training"]
        self.seq_len: int = t["seq_len"]
        self.warm_frac: float = t["warm_frac"]
        self.weight_decay: float = t["weight_decay"]
        self.grad_clip: float = t["grad_clip"]
        self.patience: int = t["patience"]
        self.bert_lr: float = t["bert_lr"]
        self.classifier_lr: float = t["classifier_lr"]
        self.rng_seed: int = t["rng_seed"]

        m = raw["model"]
        self.dropout: float = m["dropout"]

        # 由 CLI 决定
        self.backbone_name: str = "bert-base-chinese"
        self.dataset_name: str = "weibo"
        self.use_swanlab: bool = True  # 默认开，--no-swanlab 关

        # 由 resolve 决定
        self.backbone_path: str = ""
        self.corpus_dir: str = ""
        self.ckpt_root: str = ""
        self.batch_size: int = 0
        self.epochs: int = 0

    def resolve(self):
        if self.backbone_name not in self.backbones:
            raise KeyError(f"未知 backbone: {self.backbone_name}")
        if self.dataset_name not in self.datasets:
            raise KeyError(f"未知 dataset: {self.dataset_name}")

        self.backbone_path = self.backbones[self.backbone_name]
        ds = self.datasets[self.dataset_name]
        self.corpus_dir = ds["corpus_dir"]
        self.ckpt_root = os.path.join(ds["ckpt_root"], self.backbone_name.replace("/", "_"))
        self.batch_size = ds["batch_size"]
        self.epochs = ds["epochs"]
        os.makedirs(self.ckpt_root, exist_ok=True)
        return self

    def as_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v if not isinstance(v, dict) else f"<dict:{len(v)}>"
        return out


def _cli():
    p = argparse.ArgumentParser(description="NER training options")
    p.add_argument("--config", type=str, default="configs.json")
    p.add_argument("--backbone", type=str, default="bert-base-chinese")
    p.add_argument("--dataset", type=str, default="weibo")
    p.add_argument("--no-swanlab", action="store_true")
    return p.parse_args()


def load_settings() -> Settings:
    cli = _cli()
    with open(cli.config, "r", encoding="utf-8") as f:
        raw = json.load(f)

    s = Settings(raw)
    s.backbone_name = cli.backbone
    s.dataset_name = cli.dataset
    s.use_swanlab = not cli.no_swanlab
    return s.resolve()
