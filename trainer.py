import os
import logging
from typing import Dict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from metrics import NERMetrics

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        settings,
        device: torch.device,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        test_loader: DataLoader,
        id2label: Dict[int, str],
    ):
        self.model = model.to(device)
        self.s = settings
        self.device = device
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.metrics = NERMetrics()
        self.id2label = id2label

        # 两组参数分组学习率（BERT / 分类头）
        groups = [
            {"params": model.bert.parameters(), "lr": self.s.bert_lr},
            {"params": model.head.parameters(), "lr": self.s.classifier_lr},
        ]
        self.optimizer = AdamW(groups, weight_decay=self.s.weight_decay)

        total_steps = len(train_loader) * self.s.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * self.s.warm_frac),
            num_training_steps=total_steps,
        )

        self.best_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_state: Dict[str, torch.Tensor] = {}

    # ---------- 工具 ----------
    def _move(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def _tags_from_ids(self, pred_ids, gold_ids):
        pred_tags, gold_tags = [], []
        for p_row, t_row in zip(pred_ids, gold_ids):
            pt, gt = [], []
            for p_i, t_i in zip(p_row, t_row):
                if t_i == -100:
                    continue
                pt.append(self.id2label[int(p_i)])
                gt.append(self.id2label[int(t_i)])
            pred_tags.append(pt)
            gold_tags.append(gt)
        return pred_tags, gold_tags

    # ---------- 单轮 ----------
    def _train_one_epoch(self, epoch: int):
        self.model.train()
        total = 0.0
        bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.s.epochs}")
        for step, batch in enumerate(bar):
            batch = self._move(batch)
            out = self.model(**batch)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.s.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            total += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

            if self.s.use_swanlab and step % 20 == 0:
                import swanlab
                swanlab.log({
                    "train/loss": loss.item(),
                    "train/lr": self.scheduler.get_last_lr()[0],
                })
        return total / max(len(self.train_loader), 1)

    # ---------- 评估 ----------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        for batch in loader:
            batch = self._move(batch)
            out = self.model(**batch)
            total_loss += out["loss"].item()
            pred_ids = torch.argmax(out["logits"], dim=-1).cpu().numpy()
            gold_ids = batch["labels"].cpu().numpy()
            pred_tags, gold_tags = self._tags_from_ids(pred_ids, gold_ids)
            self.metrics.update(pred_tags, gold_tags)

        result = self.metrics.compute()
        result["loss"] = total_loss / max(len(loader), 1)
        result["report"] = self.metrics.report()
        return result

    # ---------- 完整流程 ----------
    def train(self):
        for ep in range(self.s.epochs):
            train_loss = self._train_one_epoch(ep)
            dev = self.evaluate(self.dev_loader)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | dev_f1=%.4f | %s",
                ep + 1, self.s.epochs, train_loss, dev["f1"], dev["report"],
            )
            if self.s.use_swanlab:
                import swanlab
                swanlab.log({
                    "epoch": ep + 1,
                    "train/epoch_loss": train_loss,
                    "dev/precision": dev["precision"],
                    "dev/recall": dev["recall"],
                    "dev/f1": dev["f1"],
                })

            if dev["f1"] > self.best_f1:
                self.best_f1 = dev["f1"]
                self.best_epoch = ep + 1
                self.patience_counter = 0
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                self._save(f"best_model_epoch_{ep + 1}")
            else:
                self.patience_counter += 1

            if (ep + 1) % 5 == 0:
                self._save(f"checkpoint_epoch_{ep + 1}")

            if self.patience_counter >= self.s.patience:
                logger.info("Early stopping at epoch %d", ep + 1)
                break

        # 恢复最优权重，再测 test
        if self.best_state:
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)
        test = self.evaluate(self.test_loader)
        self._save("final_model")
        logger.info("Test | f1=%.4f | %s", test["f1"], test["report"])
        if self.s.use_swanlab:
            import swanlab
            swanlab.log({
                "test/precision": test["precision"],
                "test/recall": test["recall"],
                "test/f1": test["f1"],
            })
            swanlab.finish()
        return test

    def _save(self, name: str):
        path = os.path.join(self.s.ckpt_root, f"{name}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
            "backbone_path": self.s.backbone_path,
            "n_tags": self.model.n_tags,
            "id2tag": self.id2label,
        }, path)
        logger.info("Saved to %s", path)
