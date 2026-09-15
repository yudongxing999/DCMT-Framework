# Paper Ablations — Fixed Protocol (DCMT / CMCE Smoke)

> **Disclaimer (必读):** 下列数字来自 CPU 烟测流水线，用于验证方法组件与复现实验协议，**不是**正式 CMCE 排行榜结果，也不可直接写成 SOTA。视觉侧监督含伪标签 / CLIP 网格接地，非人工框标注。

Related logs: [EXPERIMENTS.md](../EXPERIMENTS.md) · metrics under `results/*/metrics.json`.

---

## 1. Fixed protocol（固定后不可悄悄改）

| Item | Freeze value |
|------|----------------|
| Model default | `tiny`: d_model=64, n_heads=4, n_layers=1, d_ff=128, v_layers=1, img=64, patch=16 (~1.27M) |
| Seed | 42（除非表格注明；hard_v3 best 曾用 43） |
| Device | CPU |
| Primary metrics | `chunk_f1`; `alignment_acc` (=argmax over text for each GT visual token); `cmce_score` |
| Secondary | `threshold_acc` (cosine/score > τ) |
| Artifacts | commit `results/<run>/metrics.json`; do **not** claim from uncommitted runs |
| Data regen | images/checkpoints not in git; scripts must regenerate |

**Claim boundary**

- ✅ 可写：组件消融趋势（InfoNCE、curriculum、matched vs unmatched、宽度/样本量）
- ❌ 不可写：超越商用 VLM / 正式 CMCE 榜；把 picsum 或 CLIP 伪接地当成人类对齐上限

Export tables from committed metrics:

```bash
python scripts/export_ablation_tables.py
python scripts/export_ablation_tables.py --format latex
```

---

## 2. Ablation matrix (A0–A6)

| ID | Run folder | Factor under test | Keep fixed |
|----|------------|-------------------|------------|
| **A0** | `tiny_baseline` | no token-level InfoNCE | easy synth, tiny |
| **A1** | `tiny_align_v2` | + token InfoNCE | easy synth, tiny |
| **A2** | `tiny_hard_v1` | hard difficulty | tiny, 8ep, no curriculum |
| **A3** | `tiny_hard_v2` | + curriculum | hard data, tiny |
| **A4** | `tiny_hard_v3` | push align ≥0.5 (failed) | hard + stronger HN/ta |
| **A5** | `tiny_real_v4` | matched Flickr + CLIP | tiny, n=100, ep=8 |
| **A6a** | `tiny_real_v5` | width↑ (`small`) | Flickr+CLIP, n=100, ep=8 |
| **A6b** | `tiny_real_v6` | n/epochs↑ | Flickr+CLIP, tiny, 320/80, ep=20 |
| **A6c** | `tiny_real_coco_v2` | domain (COCO) | tiny scaled |

Optional context (not core A-ids): `tiny_real_v1–v3` (picsum grounding chain), `tiny_real_coco_v1` (COCO+small).

---

## 3. Paper-ready tables

### Table 1. Token-align loss (easy synthetic)

**Caption (EN):** Effect of token-level InfoNCE on easy synthetic CMCE (tiny, seed 42, CPU).

| Setting | alignment_acc | cmce_score | chunk_f1 |
|---------|--------------:|-----------:|---------:|
| A0 w/o token InfoNCE | 0.288 | 0.573 | ≈1.00 |
| A1 + token InfoNCE | **0.817** | **0.890** | ≈1.00 |
| Δ | +0.529 | +0.317 | — |

```latex
\begin{tabular}{lccc}
\hline
Setting & Align. & CMCE & Chunk F1 \\
\hline
w/o token InfoNCE & 0.288 & 0.573 & $\approx$1.00 \\
+ token InfoNCE & \textbf{0.817} & \textbf{0.890} & $\approx$1.00 \\
\hline
\end{tabular}
```

### Table 2. Hard synthetic + curriculum

**Caption (EN):** Hard-synthetic alignment under curriculum learning (tiny, CPU smoke).

| Setting | alignment_acc | cmce_score | Notes |
|---------|--------------:|-----------:|-------|
| A2 hard, 8ep | 0.142 | 0.483 | no curriculum |
| A3 + curriculum | 0.356 | 0.607 | ≥0.35 met |
| A4 stronger push | 0.380 | 0.612 | ≥0.5 missed |

```latex
\begin{tabular}{lccc}
\hline
Setting & Align. & CMCE & Note \\
\hline
Hard, 8 epochs & 0.142 & 0.483 & no curriculum \\
+ curriculum & 0.356 & 0.607 & target $\geq$0.35 \\
Stronger push & 0.380 & 0.612 & target $\geq$0.5 missed \\
\hline
\end{tabular}
```

### Table 3. Real matched data & negative scalings

**Caption (EN):** Matched Flickr/COCO with CLIP grid grounding; width/data scaling ablations (CPU smoke).

| Setting | Model | n (tr/va) | ep | alignment_acc | cmce_score |
|---------|-------|-----------|---:|--------------:|-----------:|
| A5 Flickr+CLIP | tiny | 80/20 | 8 | **0.320** | 0.557 |
| A6a + width (`small`) | small | 80/20 | 8 | 0.156 | 0.471 |
| A6b + n/epochs | tiny | 320/80 | 20 | 0.273 | 0.526 |
| A6c COCO scaled | tiny | 320/80 | 20 | 0.291 | 0.535 |

```latex
\begin{tabular}{llrrcc}
\hline
Setting & Model & $n$ & Ep. & Align. & CMCE \\
\hline
Flickr+CLIP & tiny & 80/20 & 8 & \textbf{0.320} & 0.557 \\
+ width (small) & small & 80/20 & 8 & 0.156 & 0.471 \\
+ $n$/epochs & tiny & 320/80 & 20 & 0.273 & 0.526 \\
COCO scaled & tiny & 320/80 & 20 & 0.291 & 0.535 \\
\hline
\end{tabular}
```

---

## 4. Suggested Results text

### 中文（可改写后进稿）

在固定 tiny 协议下，仅引入 token 级 InfoNCE 便将简单合成集上的 alignment_acc 从 0.288 提升至 0.817（表 1），说明批级对比学习不足以监督跨模态 token 对齐。提高合成难度后，无课程学习时对齐降至 0.142；加入 easy→hard 课程后回升至 0.356，进一步加强训练后达到 0.380，仍未突破 0.5（表 2），表明当前容量与难例设定下存在明显上限。在真实匹配 Flickr 子集（CLIP 网格接地）上，tiny 取得 0.320 的对齐；同数据加宽网络或扩大样本/轮次均未超过该点（表 3）。因此，后续增益更可能来自更强的视觉接地（检测/分割框），而非单纯放大宽度或烟测规模。

### English (draft)

Under a frozen tiny protocol, adding token-level InfoNCE raises easy-synthetic alignment from 0.288 to 0.817 (Table 1), showing that batch contrastive loss alone is insufficient for token alignment. On hard synthetics, alignment falls to 0.142 without curriculum and recovers to 0.356–0.380 with curriculum and stronger objectives, still below 0.5 (Table 2). On matched Flickr captions with CLIP grid grounding, tiny reaches 0.320; increasing width or data/epochs does not surpass this point (Table 3). These smoke results motivate stronger visual grounding rather than scale-only tweaks.

---

## 5. Limitations

1. CPU smoke, small \(n\), short schedules — variance and under-training possible.  
2. Visual GT is pseudo (spatial / CLIP patch grid), not human boxes.  
3. `chunk_f1≈1` on easy synth is near-saturated and weak as a differentiator.  
4. Picsum runs are pipeline checks only; do not mix with matched-Flickr claims.  
5. Hard ≥0.5 and real SOTA are **out of scope** for this log.

---

## 6. Next slot — stronger grounding / detection（未跑）

**Goal:** Replace CLIP-grid pseudo boxes with open-vocab detection/segmentation, compare under **identical** A5 protocol.

| Item | Spec |
|------|------|
| Data | Same Flickr slice as A5: n=100, seed 42, max-caption-tokens 12, tiny, ep=8 |
| Baseline | A5 CLIP grid (`tiny_real_v4`, align≈0.320) |
| Det candidate | OWL-ViT / Grounding DINO → phrase→box → map box center to patch index `v` |
| Seg candidate | CLIPSeg / similar → mask centroid → patch `v` |
| Success | `alignment_acc` > A5 on same val split; report failure if install/OOM |
| Budget | Prefer CPU-capable tiny detector or one GPU short run; document model id + checksum |
| Deliverable | `results/tiny_real_det_v1/metrics.json` + update Table 3 row |

Do **not** change model width, loss weights, or seed when comparing Det vs CLIP-grid.

---

## 7. Checklist before citing a number

- [ ] `results/<run>/metrics.json` exists on `main`  
- [ ] Protocol row matches this doc (tiny/seed/data)  
- [ ] Visual supervision type stated (synth GT / CLIP grid / detector)  
- [ ] Smoke disclaimer present in paper draft  
