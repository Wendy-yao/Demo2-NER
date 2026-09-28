import logging
import random

import numpy as np
import torch

from config import load_settings
from data import DataModule
from module import TaggingNet
from trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    s = load_settings()
    fix_seed(s.rng_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dataset=%s Backbone=%s Device=%s",
                s.dataset_name, s.backbone_name, device)

    dm = DataModule(
        corpus_dir=s.corpus_dir,
        backbone_path=s.backbone_path,
        seq_len=s.seq_len,
        bs=s.batch_size,
    ).setup()
    logger.info("num_labels=%d | id2label=%s", dm.num_labels, dm.id2label)

    model = TaggingNet(s.backbone_path, n_tags=dm.num_labels, settings=s).to(device)
    logger.info("Params: total=%d trainable=%d",
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in model.parameters() if p.requires_grad))

    if s.use_swanlab:
        import swanlab
        swanlab.init(
            project="demo2-ner",
            experiment_name=f"{s.dataset_name}_{s.backbone_name}",
            config=s.as_dict(),
        )

    trainer = Trainer(
        model=model,
        settings=s,
        device=device,
        train_loader=dm.loader("train"),
        dev_loader=dm.loader("dev"),
        test_loader=dm.loader("test"),
        id2label=dm.id2label,
    )
    test = trainer.train()
    logger.info("Final Test: %s", test["report"])


if __name__ == "__main__":
    main()
