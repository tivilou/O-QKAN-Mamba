# Phase D: 实验规划

## 提出的模型

**OPFA-QKAN-Mamba**：`y = x + sigmoid(OPFA_QKAN(x, c)) × Mamba(x)`

- 门控：OPFA-DARUAN（多轴量子编码 + 频率分区 + 自适应测量）
- 混合器：Mamba SSM（选择性状态空间模型）
- 环境：GPU（use_mamba=True）

---

## 数据集

| 数据集 | 患者数 | 特征维度 | 任务 | 角色 |
|--------|--------|----------|------|------|
| **PhysioNet 2019 Sepsis Challenge** | 40,336 | 34 | 脓毒症早期预测 | 主实验 |
| eICU Sepsis (Clinical Time Series) | 3,362 | 4 | ICU 死亡预测 | 外部验证 |
| eICU Cardiac Arrest (Clinical Time Series) | 64,589 | 4 | ICU 死亡预测 | 外部验证 |
| MIMIC GIB (Clinical Time Series) | 2,602 | 4 | 院内死亡预测 | 外部验证 |

### 主数据集特征到频段映射（PhysioNet 2019, 34维）

```
感染频段 (σ_z)：Temp, WBC, Lactate, Fibrinogen, BUN, BaseExcess
循环频段 (σ_x)：HR, MAP, SBP, DBP, Resp, O2Sat, EtCO2
器官频段 (σ_y)：Creatinine, Bilirubin_direct, Bilirubin_total, Platelets, AST, Alkalinephos
静态/人口学（不参与频段分区）：Age, Gender, HospAdmTime, ICULOS
其余实验室指标按临床归属分配到对应频段
```

---

## 主实验（PhysioNet 2019 Sepsis Challenge）

### 实验 1：基线对比（论文 Table 1）

| 组别 | 模型 | 说明 |
|------|------|------|
| 经典序列模型 | LSTM | 标准 RNN 基线 |
| 经典序列模型 | GRU | 轻量 RNN 基线 |
| 经典序列模型 | Transformer | 自注意力基线 |
| 经典序列模型 | TCN | 时序卷积基线 |
| Mamba 变体 | Mamba (pure) | 纯 Mamba，无门控 |
| Mamba 变体 | Mamba + MLP gate | Mamba + 经典 MLP 门控 |
| Mamba 变体 | Mamba + sigmoid gate | Mamba + 简单可学习门控 |
| QKAN 变体 | Original DARUAN + Mamba | 旧架构（FiLM 条件化）+ Mamba |
| **Ours** | **OPFA-QKAN-Mamba** | 完整提出的模型 |

**目的**：
- 经典基线：确定整体性能水平
- Mamba 变体：证明 QKAN 门控优于经典门控
- QKAN 变体：证明 OPFA 修改优于旧的 FiLM 条件化

### 实验 2：消融实验（论文 Table 2）

| 变体 | 去掉了什么 | 验证什么 |
|------|-----------|----------|
| Full model | 无（完整 OPFA-QKAN-Mamba） | 上界 |
| w/o multi-axis | 全部用 σ_z 编码 | 多轴编码的贡献 |
| w/o frequency partition | 不分频段，全局共享权重 | 频率分区的贡献 |
| w/o adaptive measurement | 固定 σ_z 测量 | 自适应测量的贡献 |
| w/o Mamba → Conv1d | 去掉 Mamba，用 Conv1d | Mamba 混合器的贡献 |
| w/o Mamba → LSTM | 去掉 Mamba，用 LSTM | Mamba vs LSTM 混合器 |
| w/o QKAN → MLP gate | 去掉量子门控，用 MLP | 量子门控的贡献 |

### 实验 3：谱证书评估（论文 Table 3）

| 指标 | 说明 |
|------|------|
| Certificate accuracy | 证书声明的 inactive band 是否真的对输出无贡献 |
| Violation detection rate | 违规预测被正确拦截的比例 |
| Deliberation trigger rate | 审慎推理被触发的频率 |
| Band-feature alignment | 频段贡献与对应特征重要性的相关性 |

---

## 外部验证（3 个 Clinical Time Series 数据集）

在 eICU Sepsis / Cardiac Arrest / MIMIC GIB 上验证模型泛化性。

由于这些数据集只有 4 维特征，频段分区退化为"每频段 1-2 个特征"，主要验证：
- OPFA-QKAN-Mamba 在低维场景下是否仍有竞争力
- 参数效率优势是否跨数据集保持
- 谱证书机制是否仍然有效

---

## 评估指标

| 指标 | 说明 | 适用实验 |
|------|------|----------|
| AUROC | 主指标，区分能力 | 全部 |
| AUPRC | 不平衡数据下更有意义 | 全部 |
| Utility Score | PhysioNet 2019 官方指标（早期预测奖励） | 主实验 |
| 参数量 | 模型复杂度 | 全部 |
| 推理时间 (ms) | 单样本推理延迟 | 主实验 |

---

## 训练配置

- Epochs: 50（early stopping, patience=10）
- Learning rate: 1e-3（Adam）
- Batch size: 128
- Gradient clipping: 1.0
- 设备: GPU (CUDA)
- 日志: 每 epoch 实时写入 `logs/experiment_*.log`
- 模型选择: 验证集最优 AUROC 对应的 checkpoint

---

## 实现步骤

1. 等待 PhysioNet 2019 Challenge 数据下载到 `data/` 目录
2. 编写 `.psv` 文件解析器 + 特征到频段映射
3. 适配模型：`d_model` 匹配实际特征维度，`use_mamba=True`
4. 实现所有基线模型和消融变体
5. 训练 + 评估（实时日志输出到 `logs/`）
6. 生成结果 JSON + 汇总表
7. 更新 HTML 报告

---

## 论文叙事

> "我们提出 OPFA-QKAN-Mamba，一个将医学本体知识编码进量子电路频率结构的
> Mamba 混合模型。实验 1 证明完整模型优于所有基线；Mamba 变体对比证明量子
> 门控优于经典门控；消融实验证明三个 QML 修改各有贡献；谱证书评估证明模型
> 提供了其他方法无法提供的可证明安全保证。"
