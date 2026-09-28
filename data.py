import os
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

SKIP_ID = -100


@dataclass
class Sentence:
    chars: List[str]
    tags: List[str]


class BioReader:
    """BIO 语料读取器。"""

    def __init__(self, corpus_dir: str):
        self.corpus_dir = corpus_dir

    def read(self, split: str) -> List[Sentence]:
        path = os.path.join(self.corpus_dir, f"{split}.txt")
        sents, chars, tags = [], [], []
        with open(path, "r", encoding="utf-8") as fin:
            for raw in fin:
                line = raw.strip()
                if not line:
                    if chars:
                        sents.append(Sentence(chars, tags))
                        chars, tags = [], []
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                chars.append(parts[0])
                tags.append(parts[1])
        if chars:
            sents.append(Sentence(chars, tags))
        return sents


class BioTaggingDataset(Dataset):
    """字级数据集：tokenizer 按字切分，word_ids 天然对齐。"""

    def __init__(self, sentences: List[Sentence], tokenizer, seq_len: int,
                 tag_to_id: Dict[str, int]):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tag_to_id = tag_to_id
        self.o_id = tag_to_id["O"]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        s = self.sentences[idx]
        enc = self.tokenizer(
            s.chars,
            is_split_into_words=True,
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        tag_ids, last_wid = [], None
        for wid in enc.word_ids():
            if wid is None:
                tag_ids.append(SKIP_ID)
            elif wid != last_wid:
                tag_ids.append(self.tag_to_id.get(s.tags[wid], self.o_id))
            else:
                tag_ids.append(SKIP_ID)
            last_wid = wid

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].new_tensor(tag_ids)
        return item


class DataModule:
    """构建 tokenizer / datasets / 标签表 / dataloaders。"""

    SPLITS: Tuple[str, ...] = ("train", "dev", "test")

    def __init__(self, corpus_dir: str, backbone_path: str, seq_len: int, bs: int, num_workers: int = 0):
        self.corpus_dir = corpus_dir
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_path)
        self.seq_len = seq_len
        self.bs = bs
        self.num_workers = num_workers

        self.reader = BioReader(corpus_dir)
        self._cache: Dict[str, List[Sentence]] = {}
        self.label2id, self.id2label = self._build_label_table()
        self.datasets: Dict[str, BioTaggingDataset] = {}

    def _read(self, split: str) -> List[Sentence]:
        if split not in self._cache:
            self._cache[split] = self.reader.read(split)
        return self._cache[split]

    @property
    def num_labels(self) -> int:
        return len(self.label2id)

    def _build_label_table(self):
        train_sents = self._read("train")
        uniq = {t for s in train_sents for t in s.tags}
        ordered = (["O"] if "O" in uniq else []) + sorted(uniq - {"O"})
        if not ordered:
            raise ValueError(f"{self.corpus_dir}/train.txt 中没有标签")
        l2i = {t: i for i, t in enumerate(ordered)}
        return l2i, {i: t for t, i in l2i.items()}

    def setup(self):
        for split in self.SPLITS:
            sents = self._read(split)
            self.datasets[split] = BioTaggingDataset(
                sents, self.tokenizer, self.seq_len, self.label2id
            )
        return self

    def loader(self, split: str):
        return DataLoader(
            self.datasets[split],
            batch_size=self.bs,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
