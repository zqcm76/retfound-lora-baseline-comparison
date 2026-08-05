# 阶段一 · 数据管道 + 传统基线

REFUGE2 数据管道、稳定的评测核心，以及两个**传统**基线：ImageNet 预训练的
ResNet34 分类器 + 从零训练的 UNet 分割器。它们是阶段二 LoRA-RETFound 的对照锚点。

---

## ⚠️ 本次修订说明（相对旧版）

| 编号 | 问题 | 处理 |
|---|---|---|
| C-1 | `smoke_test.py` **必然 ImportError**（按 `src/`+`scripts/` 布局算路径，实际文件平铺）；四个脚本用了三套 sys.path 约定 | 统一约定，实测可跑通 |
| C-2 | `metrics.py` 阈值写死 0.5、只回聚合值不回逐样本概率 | 阈值可传、逐样本可回、加 bootstrap CI 与校准 |
| C-3 | test 结果只存在于两个含硬编码路径与用户名的 notebook | 抽成 `evaluate.py`，落盘逐样本证据表 |
| C-4 | `Config("baseline.yaml")` 把 `cfg.data` 静默设成字符串 | `__post_init__` 直接拒绝 |
| C-5 | 标签查找失败时全部静默变成 label=0 | 抛异常并打印类别分布 |
| C-6 | 文档大篇幅宣传从未做过的 RIGA/PALM 零样本实验 | 删除，`FundusSegDataset` 标记为 NOT USED |
| C-7 | 声称做了单元测试但仓库里没有 | 补 `tests/test_metrics.py`，16 项全过 |
| C-8 | `masks.py` 灰度 64 归类错误，且并非 docstring 声称的 nearest-of-three | 改为真正的最近邻 |
| C-9 | `np.mean(dd + dc)` 是列表拼接，易被误读 | 显式写法 |
| C-10 | 依赖全部未锁版本 | 全部锁定 |
| C-11 | checkpoint 不存配置/种子/commit | 全部落盘 |

---

## 目录结构（**平铺**，与文档一致）

```
phase1/
├── baseline.yaml          超参数
├── config.py              dataclass 配置 + YAML 加载（含类型断言）
├── masks.py               REFUGE 掩膜 -> {0:bg,1:rim,2:cup}（真最近邻）
├── transforms.py          转发到 retfound_common.transforms（唯一实现）
├── datasets.py            REFUGE2Dataset（唯一训练集）
├── models.py              ResNet34Classifier / UNet / BCE / Dice-CE
├── metrics.py             AUC+CI、Dice、CDR、阈值标定、校准
├── engine.py              通用 Trainer + 评估器
├── utils.py               种子、设备、计量器、参数统计、git commit
├── make_synthetic_data.py 合成数据
├── smoke_test.py          端到端自检（真的能跑通）
├── train_classifier.py    ResNet34 训练
├── train_segmenter.py     UNet 训练
├── evaluate.py            ★ 评测入口（阈值在 val 标定，test 只报告）
└── tests/test_metrics.py  单元测试
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 1) 自检（不需要任何真实数据）
python smoke_test.py            # 应输出 ALL SMOKE TESTS PASSED

# 2) 单元测试
python -m pytest tests/ -q      # 16 passed

# 3) 训练
python train_classifier.py --config baseline.yaml --index data/data_all.csv
python train_segmenter.py  --config baseline.yaml --index data/data_all.csv

# 4) 评测（注意：阈值在 val 上标定，绝不能在 test 上拟合）
python evaluate.py --config baseline.yaml --task cls \
    --ckpt outputs/baseline_resnet34_best.pt --split test --calibrate-on val --out results/
python evaluate.py --config baseline.yaml --task seg \
    --ckpt outputs/baseline_unet_best.pt --split test --out results/
```

## 数据集

**本项目只使用 REFUGE2。** RIGA / PALM / Drishti-GS / RIM-ONE / G1020 从未被使用，
不要在任何文档中把它们描述成已完成的零样本评测（FIX(C-6)）。

REFUGE2 的真实设备构成（**这是文档里错得最久的一处**）：

| Split | 数量 | 相机 | 分辨率 |
|---|---|---|---|
| Train | 1200 | Zeiss Visucam 500 (400) **+ Canon CR-2 (800)** | 2124×2056 / 1634×1634 |
| Val | 400 | 第三台相机（未见） | 1940×1940 |
| Test | 400 | 第四台相机（未见） | 1848×1848 |

**Canon CR-2 是训练集相机，不是测试集相机。** 「Train=Zeiss / Val,Test=Canon」是
REFUGE**1** 的结构。

### ⚠️ 开训前核实掩膜编码

```python
from masks import sanity_check_encoding
from PIL import Image; import numpy as np
print(sanity_check_encoding(np.asarray(Image.open("REFUGE2/Train/mask/0001.bmp").convert("L"))))
```
默认约定 `0=cup, 128=rim, 255=bg`。不符就改 `masks.py` 顶部三个常量。

## 实测基线结果（同一 split、同为 val 选优）

| 指标 | Val | Test |
|---|---|---|
| ResNet34 AUC | 0.9310 | **0.8775** |
| ResNet34 sens / spec @0.5 | 0.600 / 0.969 | 0.850 / 0.772 |
| UNet Disc Dice | 0.9419 | 0.9287 |
| UNet Cup Dice | 0.8438 | 0.8448 |
| UNet vCDR MAE | 0.0501 | 0.0684 |

两点必须随表一起写：

1. **固定阈值 0.5 下 sens/spec 跨设备完全反转**（0.60/0.97 → 0.85/0.77），这是
   "需要设备特定校准"最直接的实验证据。
2. Phase 1 分割在 **512×512** 评测，Phase 2 在 **384×384**，Dice 不能直接比 ——
   `evaluate.py` 会把 `eval_resolution` 写进报告。
