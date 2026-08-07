# 运行顺序手册

> **一条贯穿全程的规则:test 集只解封一次。**
> 直觉上会想"训完 phase1 就评 test、训完 phase2 再评 test",这是错的 ——
> 每碰一次 test 就多一次隐性的模型选择。正确做法是:
> **Step 1-4 全部只在 val 上做决策,Step 5 一次性跑完所有 test 评测。**

---

## 依赖关系

```
Step 0  环境 + 数据 + 权重
   │
Step 0.5  自检（不碰真实数据,10 分钟）
   │
   ├──────────────┬──────────────────┐
   │              │                  │
Step 1         Step 2            Step 3
phase1 基线    phase2 主模型      phase2 LOCO
（独立）       best.pth          （独立训练自己的模型）
   │              │                  │
   │              ▼                  │
   │           Step 4                │
   │           phase3 消融           │
   │           head_*.pth            │
   │              │                  │
   └──────────────┴──────────────────┘
                  ▼
              Step 5  🔓 test 解封（一次性）
                  ▼
              Step 6  汇总成表
```

Step 1 / Step 3 与 Step 2 之间没有依赖,有多卡可以并行。
**Step 4 必须等 Step 2 出 `best.pth`,Step 5 必须等 Step 4。**

---

## Step 0 · 环境与数据

### 0.1 安装

```bash
cd RETFound-glaucoma-fixed
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# GPU 版 torch 要单独装,否则会装到 CPU 版
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r phase1/requirements.txt
pip install -r phase2/requirements.txt
pip install -r phase4/requirements.txt
```

### 0.2 数据索引 CSV

四个阶段共用同一张 `data/data_all.csv`。必需列:

| 列名 | 说明 | 可接受的别名 |
|---|---|---|
| `image_path` | 眼底图绝对/相对路径 | `image` `img_path` `path` `filepath` |
| `mask_path` | REFUGE2 灰度掩膜 (bg=255, rim=128, cup=0) | `mask` `seg_path` `gt_path` |
| `label` | 1=青光眼, 0=正常 | `cls_label` `class` `glaucoma` `y` |
| `fovea_x` / `fovea_y` | 黄斑绝对像素坐标 | `macula_x/y` `fx/fy` |
| `split` | `train` / `val` / `test` | — |
| `ID` | 可选,用于证据表回溯 | — |

> 若你的掩膜是分开的两张(视盘/视杯各一张),用 `disc_path` + `cup_path` 列代替
> `mask_path`,代码会自动走另一条分支。

### 0.3 ⚠️ 开训前核实掩膜编码(五分钟,省下几天)

```bash
python -c "
import sys; sys.path.insert(0,'phase1')
from masks import sanity_check_encoding
from PIL import Image; import numpy as np, pandas as pd
p = pd.read_csv('data/data_all.csv')['mask_path'].iloc[0]
print(sanity_check_encoding(np.asarray(Image.open(p).convert('L'))))
"
```

期望输出类似 `bg_frac≈0.97, rim_frac≈0.02, cup_frac≈0.01`。
如果 `bg_frac` 接近 0 或 `cup_frac` 超过 0.1,说明极性反了 ——
改 `phase1/masks.py` 顶部三个常量,或改
`retfound_common/dataset.py` 的 `disc_gray_max` / `cup_gray_max`。

### 0.4 RETFound 权重

把 `RETFound_mae_natureCFP.pth` 放到 `RETFound_MAE/` 下。
**注意许可证:RETFound 官方代码是 CC BY-NC 4.0,若你的 `RETFound_MAE/`
下有改自官方的文件,仓库根目录必须保留其原始 LICENSE 并注明修改。**

---

## Step 0.5 · 自检(不需要真实数据,约 10 分钟)

按顺序跑,任何一条失败就先修再往下:

```bash
python phase1/smoke_test.py                    # → ALL SMOKE TESTS PASSED
python -m pytest phase1/tests/ -q              # → 16 passed
python phase2/smoke_test.py                    # tiny 配置,CPU 可跑
python phase3/train_stage3_real.py --dummy --mode all --epochs 3
python phase4/run_phase4.py --mode synthetic --out /tmp/p4_syn
```

后两条会打印醒目的 `SYNTHETIC` 横幅 —— 那是故意的,产物不可用于任何报告。

---

## Step 1 · Phase 1 基线(可与 Step 2 并行)

```bash
cd phase1

python train_classifier.py --config baseline.yaml --index ../data/data_all.csv
python train_segmenter.py  --config baseline.yaml --index ../data/data_all.csv
```

产物:`phase1/outputs/baseline_resnet34_best.pt`、`baseline_unet_best.pt`

**此时不要评 test。** 训练过程本身只看 val。

---

## Step 2 · Phase 2 主模型(最重的一步)

```bash
cd phase2

python train.py --preset full \
    --retfound ../RETFound_MAE/RETFound_mae_natureCFP.pth \
    --manifest ../data/data_all.csv --split-col split \
    --balancer uncertainty --seg-mode coarse2fine \
    --focal-alpha 0.1,0.9 --label-smoothing 0.05 \
    --lr 2e-4 --head-lr 1e-3 --epochs 80 \
    --seed 42 --out-dir ./runs/phase2_v2
```


### 📌 顺手记录训练成本(修 U-3 需要)

训练结束后手写一份 `phase2/runs/phase2_v2/train_meta.json`:

```json
{
  "params_total_m": 315.9,
  "params_trainable_m": 13.53,
  "train_hours": 6.5,
  "peak_train_mem_gb": 11.4,
  "infer_ms_per_img": 41.0,
  "epochs": 80,
  "gpu": "RTX 3060 12GB"
}
```

峰值显存看 `nvidia-smi` 或 `torch.cuda.max_memory_allocated()/1e9`。
**这几个数必须实测**，Step 5 的 `--train-meta-json` 会读这个文件;不提供就留 NaN,绝不臆造。

### 建议同时跑 linear probe 对照(P1-5,一次运行回答"LoRA 买到了什么")

```bash
python train.py --preset full --freeze-lora \
    --retfound ../RETFound_MAE/RETFound_mae_natureCFP.pth \
    --manifest ../data/data_all.csv --epochs 80 \
    --seed 42 --out-dir ./runs/linear_probe
```

---

## Step 3 · LOCO 跨相机实验(可与 Step 2 并行)

```bash
cd phase2

python loco_runner.py \
    --manifest ../data/data_all.csv \
    --retfound ../RETFound_MAE/RETFound_mae_natureCFP.pth \
    --out-root ./runs/loco --seeds 42 123 456 --epochs 80
```

**这是整个项目里唯一真正做了跨设备量化的地方**(phase4 真实模式下只有一个设备)。

`canon_a` / `zeiss` 是**训练集**,只作 sanity check。旧版把 Zeiss 的 0.998 当成
"Canon-A→Zeiss cross device"写进了 README —— 那是训练集记忆。

---

## Step 4 · Phase 3 CDR 消融(依赖 Step 2)

```bash
cd phase3

python train_stage3_real.py \
    --stage2-ckpt ../phase2/runs/phase2_v2/best.pth \
    --manifest ../data/data_all.csv --split-col split \
    --mode all --epochs 15 --seg-size 96 \
    --focal-alpha 0.1,0.9 --label-smoothing 0.05 \
    --out-dir ./runs/phase3
```

`--mode all` 现在跑**四臂**。看输出末尾的配对 bootstrap:

```
cdr_soft − cdr_two_stage = +0.0030 [-0.0210, +0.0270]  p=0.78
```

括号里跨 0 就是未达显著。再结合 `pure_e2e_unfrozen` 那一行,才知道
"解冻分割"和"加入 CDR"各自贡献了多少。

产物:`phase3/runs/phase3/head_cdr_soft.pth` 等四个头。

---

## Step 5 · 🔓 test 解封(一次性跑完,之后不再动)

到这里所有超参、所有模型选择都已在 val 上定好。现在按顺序跑完下面五条,
**跑完就不要再改任何东西回头重跑**。

### 5.1 Phase 1 基线的 test 成绩

```bash
cd phase1
python evaluate.py --config baseline.yaml --index ../data/data_all.csv \
    --task cls --ckpt outputs/baseline_resnet34_best.pt \
    --split test --calibrate-on val --out results/

python evaluate.py --config baseline.yaml --index ../data/data_all.csv \
    --task seg --ckpt outputs/baseline_unet_best.pt \
    --split test --out results/
```

### 5.2 Phase 2 主模型的 test 成绩

```bash
cd phase2
python evaluate.py --ckpt runs/phase2_v2/best.pth \
    --manifest ../data/data_all.csv \
    --split test --calibrate-on val \
    --seg-source fine --out results/phase2_test
```

`--calibrate-on` 和 `--split` 相同会被直接拒绝执行。

### 5.3 Phase 4 最终评测 —— 带 CDR 融合

```bash
cd phase4
python run_phase4.py --mode real --out phase4_report_real \
    --csv ../data/data_all.csv \
    --stage2-ckpt ../phase2/runs/phase2_v2/best.pth \
    --stage3-head ../phase3/runs/phase3/head_cdr_soft.pth \
    --train-meta-json ../phase2/runs/phase2_v2/train_meta.json \
    --split test --calib-split val \
    --seg-source fine --device cuda
```

### 5.4 Phase 4 对照 —— 不带 CDR

```bash
python run_phase4.py --mode real --out phase4_report_nocdr \
    --csv ../data/data_all.csv \
    --stage2-ckpt ../phase2/runs/phase2_v2/best.pth \
    --split test --calib-split val --seg-source fine --device cuda
```

> 加一组 `--seg-source coarse`,就得到"coarse-only vs coarse-to-fine"的消融

---

## Step 6 · 汇总

跑完后手上会有:

| 文件 | 内容 |
|---|---|
| `phase1/results/report_{cls,seg}_test.json` | 基线 test 成绩(带 CI) |
| `phase1/results/evidence_*_test.csv` | 基线逐样本 |
| `phase2/results/phase2_test/report_test.json` | 主模型 test 成绩 |
| `phase2/runs/loco/loco_summary.csv` | 跨相机(三 seed) |
| `phase3/runs/phase3/ablation.csv` + `evidence_*.csv` | 四臂消融 |
| `phase4/phase4_report_real/REPORT.md` | 最终报告(头部声明配置) |

### 做配对检验(判断差异是否真的存在)

```bash
python - <<'PY'
import sys, pandas as pd; sys.path.insert(0, '.')
from retfound_common.stats import bootstrap_auc_delta, fmt_ci, bootstrap_auc

a = pd.read_csv('phase1/results/evidence_cls_test.csv')      # ResNet34
b = pd.read_csv('phase2/results/phase2_test/evidence_test.csv')  # LoRA-RETFound
m = a.merge(b, on='image_path', suffixes=('_r34', '_lora'))

print('ResNet34      ', fmt_ci(bootstrap_auc(m.cls_label_r34, m.cls_prob_r34)))
print('LoRA-RETFound ', fmt_ci(bootstrap_auc(m.cls_label_r34, m.cls_prob_lora)))
d = bootstrap_auc_delta(m.cls_label_r34, m.cls_prob_lora, m.cls_prob_r34)
print(f"Δ = {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  p={d['p_two_sided']:.3f}")
PY
```

---
