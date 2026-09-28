# demo2

基于 BERT 的中文命名实体识别（NER）项目，支持 `weibo` 和 `MSRA` 两套 BIO 标注数据集，可在 `bert-base-chinese` 与 `chinese-bert-wwm` 两个本地预训练模型之间切换。

## 项目结构

```text
demo2/
├── main.py              # 训练入口
├── config.py            # 命令行参数与 JSON 配置读取
├── configs.json         # backbone、数据集和训练超参数配置
├── data.py              # BIO 数据读取、标签表构建、Dataset/DataLoader
├── module.py            # BERT + Dropout + token 分类头
├── trainer.py           # 训练、验证、测试、保存 checkpoint
├── metrics.py           # 实体级 precision / recall / F1
├── predict.py           # 加载 checkpoint 进行单句预测
├── data/
│   ├── weibo/
│   │   ├── train.txt
│   │   ├── dev.txt
│   │   ├── test.txt
│   │   └── class.txt
│   └── MSRA/
│       ├── train.txt
│       ├── dev.txt
│       └── test.txt
├── models/
│   ├── bert-base-chinese/
│   └── chinese-bert-wwm/
└── results/             # 训练生成的模型权重
```

## 功能说明

- 使用 BIO 格式语料进行字级中文 NER。
- 使用 `AutoTokenizer` 对字符序列进行编码，并将标签与 token 对齐。
- 模型结构为 `BERT + Dropout + Linear`，使用 token-level 交叉熵损失。
- 支持 BERT 主干和分类头设置不同学习率。
- 使用 warmup 线性学习率调度、梯度裁剪和 early stopping。
- 以验证集实体级 F1 保存最佳模型。
- 每 5 个 epoch 保存一次 checkpoint，训练结束保存 `final_model.pt`。
- 支持 SwanLab 记录训练、验证和测试指标。
- 提供 `Predictor` 类加载 checkpoint 对单句文本进行实体标注。

## 安装依赖

建议使用 Python 3.8 环境。

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch transformers tqdm numpy swanlab
```

如果 PowerShell 禁止执行激活脚本，可以直接使用虚拟环境中的 Python：

```powershell
.\venv\Scripts\python.exe -m pip install torch transformers tqdm numpy swanlab
```

## 配置说明

主要配置写在 `configs.json` 中：

```json
{
  "backbones": {
    "bert-base-chinese": "models/bert-base-chinese",
    "chinese-bert-wwm": "models/chinese-bert-wwm"
  },
  "datasets": {
    "weibo": {
      "corpus_dir": "data/weibo",
      "ckpt_root": "./results/weibo",
      "batch_size": 16,
      "epochs": 15
    },
    "msra": {
      "corpus_dir": "data/MSRA",
      "ckpt_root": "./results/msra",
      "batch_size": 32,
      "epochs": 10
    }
  },
  "training": {
    "seq_len": 128,
    "warm_frac": 0.1,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "patience": 3,
    "bert_lr": 2e-5,
    "classifier_lr": 1e-3,
    "rng_seed": 42
  },
  "model": {
    "dropout": 0.2
  }
}
```

常用可调参数：

- `backbones`：本地预训练模型路径
- `datasets`：数据集路径、权重保存路径、batch size 和 epoch 数
- `seq_len`：最大序列长度
- `bert_lr`：BERT 主干学习率
- `classifier_lr`：分类头学习率
- `warm_frac`：warmup 比例
- `patience`：early stopping 容忍轮数
- `dropout`：分类头前的 dropout 比例

## 数据格式

训练、验证和测试文件使用 BIO 标注格式，每行包含一个字符和一个标签，中间用空白分隔；空行表示一个句子结束。

示例：

```text
我 O
爱 O
北 B-LOC
京 I-LOC

```

程序会从训练集自动构建标签表，`O` 标签会被优先放在标签 id 的第 0 位。

## 训练

默认训练 `weibo` 数据集和 `bert-base-chinese`：

```powershell
python main.py
```

指定数据集：

```powershell
python main.py --dataset msra
```

指定 backbone：

```powershell
python main.py --backbone chinese-bert-wwm
```

指定配置文件：

```powershell
python main.py --config configs.json
```

关闭 SwanLab：

```powershell
python main.py --no-swanlab
```

组合示例：

```powershell
python main.py --dataset msra --backbone chinese-bert-wwm --no-swanlab
```

## 输出文件

模型权重会保存到：

```text
results/<dataset>/<backbone>/
```

例如：

```text
results/weibo/bert-base-chinese/
├── best_model_epoch_1.pt
├── best_model_epoch_2.pt
├── checkpoint_epoch_5.pt
└── final_model.pt
```

保存的 checkpoint 中包含：

- `model_state_dict`：模型参数
- `best_f1`：当前最佳验证集 F1
- `best_epoch`：最佳 epoch
- `backbone_path`：预训练模型路径
- `n_tags`：标签数量
- `id2tag`：标签 id 到标签名的映射

## 评估指标

`metrics.py` 使用实体级别的 precision、recall、F1：

- 从 BIO 标签序列中抽取实体 `(type, start, end)`
- 预测实体与真实实体完全匹配才算正确
- 统计 TP、预测实体数、真实实体数
- 计算 precision、recall、F1

训练过程中会在每个 epoch 后输出验证集指标；训练结束后会加载最佳权重并在测试集上评估。

## 预测

`predict.py` 提供 `Predictor` 类，可以加载 `.pt` 权重进行单句预测。

示例：

```python
from predict import Predictor

predictor = Predictor("results/weibo/bert-base-chinese/final_model.pt")
print(predictor.tag("我爱北京天安门"))
```

输出格式为：

```text
[("我", "O"), ("爱", "O"), ("北", "B-LOC"), ...]
```

## SwanLab

默认启用 SwanLab，项目名为：

```text
demo2-ner
```
