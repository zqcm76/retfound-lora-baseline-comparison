# -*- coding: utf-8 -*-
"""
CPU 冒烟测试: 用 tiny_config 几秒钟跑完，验证关键风险点。

覆盖:
  1. uncertainty 路径前向 + 反传，loss 有限
  2. 冻结正确性: 只有 lora/heads/balancer 有梯度; backbone 非 LoRA 参数 grad 为 None;
     patch_embed.proj.weight.requires_grad == False
  3. GradNorm 路径双反传跑通
  4. grad_checkpointing=True 的反传路径
  5. 位置编码插值: 在更大 img 上构建模型并加载较小 grid 的伪权重，pos_embed 形状被插值
  6. evaluate() 跑通并出指标
"""
from __future__ import annotations

import sys
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401

from retfound_common.backbone import build_backbone, interpolate_pos_embed
from retfound_common.balancing import GradNorm, UncertaintyWeighting
from retfound_common.config import (BackboneCfg, format_param_breakdown,
                                    param_breakdown, tiny_config)
from retfound_common.dataset import DummyMultiTaskDataset, multitask_collate
from retfound_common.model import MultiTaskModel
from retfound_common.trainer import Trainer


def banner(msg):
    print("\n" + "=" * 60 + f"\n{msg}\n" + "=" * 60)


def make_loader(cfg, n=4):
    H, W = cfg.backbone.img_size
    ds = DummyMultiTaskDataset(n=n, img_size=(H, W),
                               num_classes=cfg.cls.num_classes, seed=1)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                      num_workers=0, collate_fn=multitask_collate)


def test_uncertainty_and_freezing():
    banner("TEST 1+2: uncertainty 前向反传 + 冻结正确性")
    cfg = tiny_config()
    device = torch.device("cpu")
    model = MultiTaskModel(cfg).to(device)
    balancer = UncertaintyWeighting(list(cfg.task_names)).to(device)
    params = model.trainable_parameters() + list(balancer.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer = Trainer(model, opt, balancer, scaler, cfg, device)

    # FIX(P0-2): 参数量必须分解到 LoRA / pos_embed / 任务头，可复算
    bd = param_breakdown(model)
    print("param breakdown:\n" + format_param_breakdown(bd))
    assert bd["trainable"] == bd["lora"] + bd["adapter"] + bd["pos_embed"] + bd["heads"], \
        "参数分解不闭合"

    # 冻结检查
    assert model.backbone.vit.patch_embed.proj.weight.requires_grad is False, \
        "patch_embed.proj 不应可训练"
    # 找一个非 LoRA 的 block 内 Linear base 权重，应冻结
    blk0 = model.backbone.vit.blocks[0]
    assert blk0.attn.qkv.base.weight.requires_grad is False, \
        "qkv.base 应冻结"
    assert blk0.attn.qkv.lora_A.requires_grad is True, "lora_A 应可训练"

    # 跑 2 个 iter
    loader = make_loader(cfg, n=4)
    it = 0
    for batch in loader:
        images = batch["image"].to(device)
        gt_disc = batch["disc_mask"].to(device)
        opt.zero_grad(set_to_none=True)
        out = model(images, gt_disc=gt_disc, use_gt_roi=True)
        losses = trainer.compute_losses(out, batch)
        total, logs = balancer(losses)
        assert torch.isfinite(total), "total loss 非有限"
        total.backward()

        # 梯度检查: backbone 非 LoRA 参数 grad 必须 None
        for name, p in model.backbone.vit.named_parameters():
            is_lora = (".lora_" in name) or (".adapter." in name)
            is_pos = name.endswith("pos_embed")
            if (not is_lora) and (not is_pos):
                assert p.grad is None, f"冻结参数有梯度: {name}"
        # LoRA 参数应有梯度
        assert blk0.attn.qkv.lora_A.grad is not None, "lora_A 无梯度"
        opt.step()
        it += 1
        print(f"  iter {it}: losses={{" +
              ", ".join(f'{k}:{float(v):.3f}' for k, v in losses.items()) + "}}")
        if it >= 2:
            break
    print("  [OK] 冻结与梯度行为正确")


def test_gradnorm():
    banner("TEST 3: GradNorm 双反传")
    cfg = tiny_config()
    cfg.balance.method = "gradnorm"
    cfg.train.accum_steps = 1
    device = torch.device("cpu")
    model = MultiTaskModel(cfg).to(device)
    balancer = GradNorm(list(cfg.task_names), alpha=cfg.balance.gradnorm_alpha).to(device)
    params = model.trainable_parameters() + list(balancer.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer = Trainer(model, opt, balancer, scaler, cfg, device)

    loader = make_loader(cfg, n=2)
    stats = trainer.train_one_epoch(loader, epoch=0)
    print("  gradnorm epoch stats:", {k: round(v, 3) for k, v in stats.items()})
    print("  weights:", balancer.logs())
    for v in stats.values():
        assert v == v, "GradNorm 出现 NaN"  # NaN != NaN
    print("  [OK] GradNorm 双反传跑通")


def test_grad_checkpointing():
    banner("TEST 4: grad_checkpointing 反传路径")
    cfg = tiny_config()
    cfg.backbone.grad_checkpointing = True
    device = torch.device("cpu")
    model = MultiTaskModel(cfg).to(device)
    model.train()
    x = torch.randn(2, 3, *cfg.backbone.img_size)
    out = model(x, use_gt_roi=False)
    loss = out["cls_logits"].sum() + out["coarse_logits"].sum() + \
        out["fine_logits"].sum() + out["macula_heatmap"].sum()
    loss.backward()
    # 确认有梯度流到 LoRA
    g = model.backbone.vit.blocks[0].attn.qkv.lora_A.grad
    assert g is not None and torch.isfinite(g).all(), "ckpt 路径梯度异常"
    print("  [OK] gradient checkpointing 反传正常")


def test_pos_embed_interp():
    banner("TEST 5: 位置编码插值")
    # 直接测函数
    pe = torch.randn(1, 1 + 8 * 8, 16)   # 8x8 grid + 1 cls
    new = interpolate_pos_embed(pe, (10, 10), num_extra_tokens=1)
    assert new.shape == (1, 1 + 10 * 10, 16), f"插值形状错: {new.shape}"
    print(f"  函数级: {tuple(pe.shape)} -> {tuple(new.shape)}  [OK]")

    # 端到端: 构建 img=160 (grid 10x10) 的 backbone，喂入 8x8 grid 的伪 state_dict
    cfg = tiny_config()
    cfg.backbone.img_size = (160, 160)   # 160/16 = 10
    bb = build_backbone(cfg.backbone)
    n_extra = bb.num_prefix_tokens
    # 伪权重: 用一个 img=128 (grid 8x8) 的同结构模型的 state_dict
    small_cfg = BackboneCfg(**{**cfg.backbone.__dict__})
    small_cfg.img_size = (128, 128)
    small = build_backbone(small_cfg)
    fake_state = {"model": small.vit.state_dict()}
    # 存成临时文件再加载 (走 load_pretrained 全流程)
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "fake_retfound.pth")
    torch.save(fake_state, tmp)
    info = bb.load_pretrained(tmp)
    print("  load info:", info)
    gh, gw = bb.grid_size
    assert bb.vit.pos_embed.shape[1] == n_extra + gh * gw, \
        "加载后 pos_embed 未插值到目标 grid"
    print(f"  端到端: 加载 8x8 权重到 10x10 模型, "
          f"pos_embed={tuple(bb.vit.pos_embed.shape)}  [OK]")


def test_checkpoint_roundtrip():
    banner("TEST 7: checkpoint 存取 (FIX(S-2))")
    import os, tempfile
    from retfound_common.checkpoint import (build_model_from_checkpoint,
                                            save_checkpoint)
    cfg = tiny_config()
    model = MultiTaskModel(cfg)
    tmp = os.path.join(tempfile.gettempdir(), "smoke_ckpt.pth")
    save_checkpoint(tmp, model, cfg, epoch=0, metrics={"auc": 0.5})
    m2, cfg2, ck = build_model_from_checkpoint(tmp, MultiTaskModel)
    assert "cfg" in ck and isinstance(ck["cfg"], dict), "checkpoint 未存完整配置"
    assert cfg2.seg.fine_out == cfg.seg.fine_out, "配置未正确恢复"
    assert cfg2.cls.label_smoothing == cfg.cls.label_smoothing
    print(f"  [OK] 配置往返一致 (fine_out={cfg2.seg.fine_out}, "
          f"label_smoothing={cfg2.cls.label_smoothing})")

    # 严格加载：故意破坏一个键，必须抛异常而不是静默保持随机初始化
    from retfound_common.checkpoint import load_model_state
    bad = {k: v for k, v in model.state_dict().items() if "cls_head" not in k}
    try:
        load_model_state(MultiTaskModel(cfg), bad, allow_partial=False)
        raise SystemExit("[FAIL] 缺键时未抛异常 —— FIX(S-2) 未生效")
    except RuntimeError as e:
        assert "missing" in str(e)
        print("  [OK] 缺键时严格拒绝，不再静默保持随机初始化")


def test_evaluate():
    banner("TEST 6: evaluate() 出指标")
    cfg = tiny_config()
    device = torch.device("cpu")
    model = MultiTaskModel(cfg).to(device)
    balancer = UncertaintyWeighting(list(cfg.task_names)).to(device)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer = Trainer(model, opt, balancer, scaler, cfg, device)
    loader = make_loader(cfg, n=4)
    metrics = trainer.evaluate(loader)
    print("  metrics:", {k: round(v, 4) for k, v in metrics.items()})
    assert "dice_mean" in metrics, "缺 dice 指标"
    assert "macula_err" in metrics, "缺 macula 指标"
    print("  [OK] evaluate 跑通")

    # FIX(S-3): 逐样本证据表必须可回 —— bootstrap CI / 阈值标定的前提
    m2, ev = trainer.evaluate(loader, return_evidence=True, threshold=0.5,
                              seg_source="fine")
    assert len(ev) == 4, f"证据表行数不对: {len(ev)}"
    for c in ("cls_prob", "cls_label", "dice_disc", "macula_err_img"):
        assert c in ev.columns, f"证据表缺列 {c}"
    print(f"  [OK] 逐样本证据表 {ev.shape}, 列={list(ev.columns)}")

    # FIX(P0-6): 两个黄斑口径必须分开命名
    assert "macula_err_img" in m2 and "macula_err_disc" in m2, \
        "黄斑两个口径未分开 —— 这正是 0.013 与 0.341 被混比的根源"
    print(f"  [OK] macula_err_img={m2['macula_err_img']:.4f} (单位=图像边长) | "
          f"macula_err_disc={m2['macula_err_disc']:.4f} (单位=视盘直径)")

    # FIX(U-9): fine 与 coarse 是两个不同口径，必须显式选择
    m_c = trainer.evaluate(loader, seg_source="coarse")
    print(f"  [OK] seg_source 可选: fine dice={m2['dice_mean']:.4f} vs "
          f"coarse dice={m_c['dice_mean']:.4f} (两者不可比，报告需注明)")


if __name__ == "__main__":
    torch.manual_seed(0)
    try:
        test_uncertainty_and_freezing()
        test_gradnorm()
        test_grad_checkpointing()
        test_pos_embed_interp()
        test_evaluate()
        test_checkpoint_roundtrip()
    except AssertionError as e:
        print("\n[FAIL]", e)
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\n[ERROR]", e)
        sys.exit(2)
    banner("ALL SMOKE TESTS PASSED ✔")
