# Echelon SEMNET 概念层设计规格书 v1.0

> **版本**: 1.0  
> **日期**: 2026-05-23  
> **状态**: 已批准  
> **作者**: 李海森 + Claude  
> **项目**: Echelon — 企业级科研趋势预测引擎

---

## 1. 项目背景与目标

### 1.1 项目定位

Echelon 是面向舜宇光学（Sunny Optical）的论文卡点分析与技术发展方向预测引擎。目标用户包括战略部门、研发部门、市场部门、总裁办等决策层。

### 1.2 本次设计范围

在 V14-B 现有三路融合架构基础上，新增 **SEMNET 概念级语义网络**作为第 4 路预测轨道。

- **V14-B 已有的三路预测**：SPC Main Path（结构延伸）+ VGAE（论文级链接预测）+ Limitation Tracking（文本驱动）
- **本次新增**：SEMNET 概念级链接预测（S5d 概念提取 + S5e 概念网络预测）
- **融合升级**：三路融合 → 四路 Meta-learning Stacking 融合

### 1.3 核心价值

| 维度 | 论文级预测（已有） | 概念级预测（新增） |
|------|-------------------|-------------------|
| 粒度 | 具体论文之间的引用关系 | 抽象概念之间的研究组合 |
| 时间视野 | 短期（1-2 年） | 中长期（2-5 年） |
| 可解释性 | 哪些论文会被引用 | 哪些课题对将被联合研究 |
| 独特能力 | 实证性验证 | 前瞻性预测（论文发表前即可预测方向） |

**文献依据**：Impact4Cast (Gu & Krenn, ML: Science and Technology 2025, AUC=0.948) 证明概念级预测可在论文发表前预测研究方向。

### 1.4 数据规模

| 阶段 | 论文数量 | 预计概念数 | 概念对数 |
|------|---------|-----------|---------|
| 验证期 | 200 篇 | ~500 | ~2,000 |
| 第一期 | 20 万篇（光学 5.4 万 + AI 14 万） | ~15,000 | ~500,000 |
| 生产期 | 100 万篇+ | ~50,000+ | ~5,000,000+ |

---

## 2. 架构概览

### 2.1 V15 流水线全景

```
独立数据采集层
────────────────────────────────────────
S0 数据爬取（OpenAlex API）→ S1 数据入库 + 嵌入计算

论文级分析层（继承 V14-B 设计）
────────────────────────────────────────
S2 SPC Main Path → S3 KeystoneScore → S4 子图构建
→ S5a SciBERT 引用分类

         ┌──── 论文级预测 ──────────────┐   ┌── 概念级预测（★ SEMNET）──┐
         │  S5b  VGAE Link Prediction  │   │  S5d  概念提取            │
         │  S5c  Limitation Tracking   │   │  S5e  概念网络+链接预测    │
         └─────────────────────────────┘   └──────────────────────────┘
                        │                              │
                        ▼              ▼               ▼              ▼
              ┌─────────────────────────────────────────────────────────┐
              │     S6 四路 Meta-learning Stacking 融合                  │
              │  Main Path + VGAE + Limitation + SEMNET → 高可信度方向   │
              └─────────────────────────────────────────────────────────┘
                                       │
                        S7 三色突变 → S8 UMAP-3D → S9 报告生成
```

### 2.2 数据自主采集原则

本项目从数据爬取开始独立构建完整流水线，不依赖 V14-B 的现有数据库。V14-B 的代码和设计作为参考蓝本，核心逻辑在新项目中重新实现。

| 原则 | 实现方式 |
|------|---------|
| 数据自主 | S0 从 OpenAlex API 独立爬取论文数据（title、abstract、引用关系、元数据），存入项目自有 SQLite |
| 嵌入自算 | S1 入库后计算 all-mpnet-base-v2 embedding，供后续所有步骤使用 |
| 统一存储 | 所有表（论文、概念、共现、预测）写入同一个 `echelon.sqlite3` 数据库 |
| 代码独立 | 新项目 `echelon/` 模块独立实现，参考但不导入 V14-B 代码 |
| 设计继承 | V14-B 的 9 步流水线设计作为蓝本，核心算法（SPC、VGAE、Limitation）在新代码中重新实现 |

---

## 3. S0 数据爬取 + S1 数据入库

### 3.1 S0 数据爬取

- **数据源**：OpenAlex API（主）+ Semantic Scholar API（补充）
- **爬取策略**：按 topic/concept 过滤，游标分页（cursor pagination），参考 V14-B 的 `fetch_pilot_papers_v2.py`
- **目标字段**：title, abstract, publication_year, cited_by_count, referenced_works, topics/concepts, authorships
- **存储格式**：原始 JSON → 清洗后写入 SQLite

### 3.2 S1 数据入库 + 嵌入计算

- **数据模型**：Pydantic v2 Paper schema（参考 V14-B 的 `echelon/schema/paper.py`），ULID 主键
- **入库表**：`paper_identity`（论文基本信息）+ `paper_references`（引用关系）
- **嵌入计算**：all-mpnet-base-v2 对 title+abstract 计算 768D embedding，写入 `paper_embeddings` 表
- **数据库**：`echelon.sqlite3`（项目自有，非 V14-B 数据库）

### 3.3 数据规模控制

| 阶段 | 爬取范围 | 论文数量 |
|------|---------|---------|
| 验证期 | 光学领域 4 个 topic（参考 V14-B CONFIG.md） | 200 篇 |
| 第一期 | 光学全领域 + AI 全领域 | 20 万篇 |
| 生产期 | 通用领域 | 100 万篇+ |

---

## 4. S5d 概念提取流水线

### 4.1 双通道提取

```
论文 title + abstract
         │
    ┌─────┴─────┐
    ▼            ▼
LLM 主通道    KeyBERT 校验通道
    │            │
    ▼            ▼
 概念集 A     概念集 B
    │            │
    └─────┬─────┘
          ▼
    去重 + 合并（Union）
          │
          ▼
    同义词归一化（嵌入聚类）
          │
          ▼
    concept_nodes + paper_concepts 表
```

#### 4.1.1 LLM 主通道

- **输入**：论文 title + abstract
- **方法**：结构化 prompt 提取所有领域相关概念（非仅关键词）
- **LLM 选择**：复用 V14-B 已有的多 LLM Provider（Anthropic Claude / OpenAI GPT-4o / Ollama 本地），通过 `echelon/v14b/config.py` 中的 provider 配置切换
- **每篇输出**：10-20 个概念短语
- **优势**：语义理解深，可提取隐含概念和方法论术语

#### 4.1.2 KeyBERT 校验通道

- **输入**：论文 title + abstract
- **方法**：all-mpnet-base-v2 嵌入 + MMR 去重（diversity=0.7）
- **每篇输出**：top-10 关键短语
- **优势**：快速、确定性、捕获 LLM 可能遗漏的表面关键词

#### 4.1.3 合并策略

- LLM 与 KeyBERT 提取的概念取并集
- 重复概念通过嵌入余弦相似度 ≥ 0.90 判定为同一概念
- 保留两个通道各自的置信度分数

**文献依据**：
- Marwitz et al. (Nature Machine Intelligence 2026) — LLM 概念提取 + 概念图谱预测研究方向，端到端验证此路径
- Dagdelen et al. (Nature Communications 2024) — LLM 微调仅需 ~500 标注样本即可高精度提取科学结构化信息
- ConExion (Norouzi et al., NSLP 2025 Workshop) — LLM 在全概念提取任务 F1 超越 KeyBERT 等 SOTA

### 4.2 同义词归一化

**方案**：纯嵌入聚类（SciBERT/Sentence-BERT 嵌入 + 余弦阈值聚类）

- **嵌入模型**：all-mpnet-base-v2（S1 入库时已计算的论文嵌入可复用）
- **聚类方法**：层次聚类（Agglomerative Clustering），cosine distance threshold = 0.15（即相似度 ≥ 0.85）
- **规范名选择**：每个聚类中出现频率最高的概念作为 canonical_name
- **前处理**：小写化 + lemmatization（spaCy）去除词形变化

**选择理由**：成熟技术，无需额外文献支撑，且在验证阶段（200 篇）足够有效。后续若需更精细的归一化，可在 Phase 2 引入 LLM 精判层。

### 4.3 输出

| 表名 | 字段 | 说明 |
|------|------|------|
| `concept_nodes` | concept_id (TEXT PK), canonical_name (TEXT), first_seen_year (INT), total_papers (INT), embedding (BLOB) | 归一化后的概念节点 |
| `paper_concepts` | paper_id (TEXT), concept_id (TEXT), score (REAL), method (TEXT) | 论文-概念关联，method 标记来源（llm/keybert） |

---

## 5. S5e 概念网络 + 链接预测

### 5.1 概念共现构建

```
paper_concepts 表
       │
       ▼
按年份切片（year_window = 1 年）
       │
       ▼
同一篇论文中共现的概念对 → 共现边
       │
       ▼
concept_cooccurrence 表（concept_a, concept_b, year, weight）
```

- **共现定义**：概念 A 和概念 B 出现在同一篇论文的 title+abstract 中
- **边权计算**：该年度共现次数 × 时间衰减因子（λ=0.8^(current_year - year)）
- **年份切片**：每个年份一个快照，保留时序演化信息

**文献依据**：Krenn & Zeilinger (PNAS 2020, 被引 200+) — SEMNET 方法论原始论文，75 万篇论文验证概念共现网络可预测研究趋势。

### 5.2 共现特征作为 KGE 输入

**关键设计修改**：共现不独立成网络，而是作为特征输入知识图谱嵌入（KGE）。

每个概念对 (A, B) 计算以下特征（基于 Kulkarni 论文 15 特征扩展）：

#### 5.2.1 基础拓扑特征（15 个，Kulkarni 原始）

| 编号 | 特征 | 计算窗口 |
|------|------|---------|
| 1-3 | degree(A) | t-2, t-1, t |
| 4-6 | degree(B) | t-2, t-1, t |
| 7-9 | shared_neighbors(A, B) | t-2, t-1, t |
| 10-12 | papers_containing_both(A, B) | t-2, t-1, t |
| 13-15 | papers_containing_either(A, B) | t-2, t-1, t |

#### 5.2.2 扩展拓扑特征（15+ 个）

| 特征组 | 包含特征 | 来源 |
|--------|---------|------|
| 中心性 | betweenness(A), closeness(A), pagerank(A) 及对称 B | Impact4Cast (2025) |
| 引用影响力 | avg_citation(A 关联论文), citation_growth_rate(A) | Impact4Cast 141 特征体系 |
| 时序趋势 | degree_growth_rate(A), degree_acceleration(A) | Gu & Krenn (ML:ST 2025) |
| 语义距离 | embedding_cosine(A, B) | 概念嵌入向量 |
| 社区结构 | same_community(A, B), community_bridge_score | Leiden 社区检测 |

#### 5.2.3 KGE 嵌入特征

- **方法**：TransE / DistMult 在概念共现图上训练嵌入
- **共现作为输入**：共现频率、时序模式、拓扑位置作为 KGE 的辅助特征
- **输出**：每个概念的 KGE 嵌入向量（128D），概念对的嵌入距离/点积作为额外预测特征

**文献依据**：
- NoGE (Nguyen et al., WSDM 2022) — 共现+KGE 结合在 CoDEx 基准达 SOTA
- FastKGE (Liu et al., IJCAI 2024) — 增量 LoRA 处理动态 KG，训练时间减少 34-49%
- Zhu et al. (EMNLP 2024 Findings) — KGE 模型间预测冲突率 8-39%，必须用集成方法

### 5.3 链接预测模型

```
全部特征向量（拓扑 + KGE 嵌入 + 时序）
              │
     ┌────────┼────────┐
     ▼        ▼        ▼
  XGBoost  LightGBM    NN
     │        │        │
     └────────┼────────┘
              ▼
       加权集成（GBDT 权重更高）
              │
              ▼
     predicted_concept_links 表（top-200）
```

#### 5.3.1 GBDT 主模型

- **实现**：XGBoost + LightGBM 集成
- **超参考范围**：max_depth=6, n_estimators=500, learning_rate=0.05
- **特征重要性**：SHAP 可解释性，输出每个预测的关键驱动特征

#### 5.3.2 前馈 NN 辅助模型

- **架构**：Kulkarni 原始架构 — 15→100→100→10→1（ReLU + Sigmoid）
- **扩展**：输入维度从 15 扩展到 30+（含扩展特征和 KGE 嵌入）
- **训练**：Adam optimizer, lr=0.001, batch_size=512, epochs=100

#### 5.3.3 集成策略

- GBDT 权重 0.6 + NN 权重 0.4（基于 Kulkarni 论文和 Science4Cast 竞赛经验）
- 最终输出 top-200 预测概念对

**文献依据**：
- Gu & Krenn (ML: Science and Technology 2025) — 2100 万篇论文验证，AUC=0.948
- Lu (IEEE BigData 2022, Science4Cast 冠军) — GBDT+GNN 混合方案效果最优
- TGB-Seq (ICLR 2025) — 时序 GNN 在复杂序列上不稳定，GBDT 更鲁棒

### 5.4 训练与评估策略

#### 5.4.1 时序 Hold-out

```
训练集：year ≤ T-3 的共现边作为正样本
验证集：year = T-2 的新出现边
测试集：year = T-1 和 T 的新出现边
负样本：随机采样未连接概念对（正负比 1:5）
```

#### 5.4.2 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| AUC-ROC | 整体判别能力 | ≥ 0.85 |
| AP (Average Precision) | 正样本排序质量 | ≥ 0.30 |
| Precision@200 | top-200 预测中的命中率 | ≥ 0.15 |
| NDCG@100 | 排序质量 | ≥ 0.40 |

### 5.5 输出

| 表名 | 字段 | 说明 |
|------|------|------|
| `concept_cooccurrence` | concept_a (TEXT), concept_b (TEXT), year (INT), weight (REAL) | 概念对共现记录 |
| `predicted_concept_links` | concept_a (TEXT), concept_b (TEXT), predicted_prob (REAL), model (TEXT), features_json (TEXT) | 预测结果，features_json 含 SHAP 值 |

---

## 6. S6 四路 Meta-learning Stacking 融合

### 6.1 融合架构

```
轨道 1: Main Path 末端延伸       → 特征向量 f1
轨道 2: VGAE 论文级预测 top-200   → 特征向量 f2
轨道 3: Limitation 未解决 top-50  → 特征向量 f3
轨道 4: SEMNET 概念对预测 top-200 → 特征向量 f4（★ 新增）
              │
              ▼
    [f1, f2, f3, f4] → RF Meta-learner → 最终排序
              │
              ▼
       future_directions 表（高可信度未来方向）
```

### 6.2 Meta-learning Stacking 设计

采用 Random Forest 作为元学习器：

1. **基学习器输出**：每个轨道对每个候选方向输出置信度分数
2. **元特征构建**：4 个轨道的置信度 + 轨道间一致性指标 + 网络属性特征（度分布、三角密度）
3. **RF 元学习器**：基于元特征学习最优融合权重，自适应不同网络特性
4. **训练数据**：历史时序中已实现的趋势作为正样本

### 6.3 概念→论文映射

SEMNET 预测的是概念对，需要映射回论文级别以与其他三路对齐：

```
SEMNET 预测：概念 A + 概念 B 将被联合研究
     │
     ▼
通过 paper_concepts 表查找：
  - 与概念 A 相关的论文集合 Pa
  - 与概念 B 相关的论文集合 Pb
     │
     ▼
Pa ∩ Pb 中的论文 = 已在研究此组合的前沿论文
Pa ∪ Pb - Pa ∩ Pb = 潜在跨领域桥接论文
     │
     ▼
与 VGAE 预测的论文链接取交集 → 高可信度
```

**文献依据**：
- Ghasemian et al. (PNAS 2020, 203 算法 × 550 网络) — 无单一算法在所有网络最优，Meta-learning Stacking 接近最优
- Zhu et al. (EMNLP 2024 Findings) — KGE 模型间预测冲突率 8-39%，集成方法减少冲突 66-78%

---

## 7. 数据库 Schema

所有表写入项目自有数据库 `echelon.sqlite3`。

### 7.1 基础表（S0/S1 数据采集层）

```sql
-- S0/S1 论文数据
CREATE TABLE paper_identity (
    paper_id        TEXT PRIMARY KEY,   -- ULID
    openalex_id     TEXT,               -- OpenAlex Work ID
    title           TEXT NOT NULL,
    abstract        TEXT,
    publication_year INTEGER,
    cited_by_count   INTEGER DEFAULT 0,
    source           TEXT,              -- 期刊/会议名
    topics_json      TEXT,              -- OpenAlex topics JSON
    created_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE paper_references (
    citing_id  TEXT NOT NULL,           -- 关联 paper_identity.paper_id
    cited_id   TEXT NOT NULL,           -- 关联 paper_identity.paper_id
    PRIMARY KEY (citing_id, cited_id)
);

CREATE TABLE paper_embeddings (
    paper_id   TEXT PRIMARY KEY,        -- 关联 paper_identity.paper_id
    embedding  BLOB NOT NULL            -- all-mpnet-base-v2 768D
);
```

### 7.2 概念表（S5d/S5e SEMNET 层）

```sql
CREATE TABLE concept_nodes (
    concept_id    TEXT PRIMARY KEY,   -- ULID
    canonical_name TEXT NOT NULL,     -- 归一化后的概念名
    first_seen_year INTEGER,         -- 首次出现年份
    total_papers   INTEGER DEFAULT 0, -- 关联论文总数
    embedding      BLOB              -- 概念嵌入向量（768D）
);

CREATE TABLE paper_concepts (
    paper_id    TEXT NOT NULL,        -- 关联 paper_identity.paper_id
    concept_id  TEXT NOT NULL,        -- 关联 concept_nodes.concept_id
    score       REAL,                -- 提取置信度
    method      TEXT,                -- 'llm' | 'keybert' | 'both'
    PRIMARY KEY (paper_id, concept_id)
);

-- S5e 共现网络 + 预测输出
CREATE TABLE concept_cooccurrence (
    concept_a  TEXT NOT NULL,         -- 关联 concept_nodes.concept_id
    concept_b  TEXT NOT NULL,         -- 关联 concept_nodes.concept_id
    year       INTEGER NOT NULL,      -- 共现年份
    weight     REAL,                  -- 加权共现次数
    PRIMARY KEY (concept_a, concept_b, year)
);

CREATE TABLE predicted_concept_links (
    concept_a      TEXT NOT NULL,
    concept_b      TEXT NOT NULL,
    predicted_prob REAL,              -- 预测概率
    model          TEXT,              -- 'xgboost' | 'lightgbm' | 'nn' | 'ensemble'
    features_json  TEXT,              -- SHAP 特征重要性 JSON
    created_at     TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (concept_a, concept_b, model)
);
```

### 7.3 索引

```sql
CREATE INDEX idx_paper_concepts_paper ON paper_concepts(paper_id);
CREATE INDEX idx_paper_concepts_concept ON paper_concepts(concept_id);
CREATE INDEX idx_cooccurrence_year ON concept_cooccurrence(year);
CREATE INDEX idx_predicted_prob ON predicted_concept_links(predicted_prob DESC);
```

---

## 8. 代码结构

全新统一架构。V14-B 的核心算法逻辑（SPC Main Path、VGAE、Limitation Tracking、KeystoneScore 等）被吸收重新实现，不保留 V14-B 原有目录结构。

```
echelon/
├── __init__.py
├── config.py                        # 全局配置（数据库路径、模型超参、API keys）
├── schema/
│   ├── __init__.py
│   └── paper.py                     # Pydantic v2 Paper 模型（ULID PK）
├── ingest/                          # S0 + S1 数据采集层
│   ├── __init__.py
│   ├── fetcher.py                   # OpenAlex API 游标分页爬取
│   ├── ingest.py                    # JSONL/JSON → SQLite 入库
│   └── embedder.py                  # all-mpnet-base-v2 嵌入计算
├── graph/                           # S2-S4 论文级图谱层（吸收 V14-B 核心逻辑）
│   ├── __init__.py
│   ├── citation_graph.py            # 引文图构建（cite_direct/co_citation/bib_couple）
│   ├── main_path.py                 # SPC Main Path 主干道分析
│   ├── keystone_score.py            # KeystoneScore 关键论文评分
│   └── subgraph.py                  # 子图构建
├── predict/                         # S5a-S5c 论文级预测层（吸收 V14-B 核心逻辑）
│   ├── __init__.py
│   ├── citation_classify.py         # SciBERT 引用功能分类
│   ├── vgae.py                      # VGAE 论文级链接预测
│   └── limitation.py                # Limitation Tracking
├── semnet/                          # S5d-S5e 概念级预测层（★ 核心新增）
│   ├── __init__.py
│   ├── concept_extract.py           # LLM + KeyBERT 双通道概念提取
│   ├── concept_normalize.py         # 嵌入聚类同义词归一化
│   ├── concept_network.py           # 共现网络构建 + 特征工程
│   ├── feature_engineering.py       # 概念对 30+ 特征计算
│   └── link_predict.py              # GBDT + NN 概念级链接预测
├── fusion/                          # S6 四路融合层
│   ├── __init__.py
│   └── stacking.py                  # Meta-learning Stacking（RF 元学习器）
├── output/                          # S7-S9 输出层
│   ├── __init__.py
│   ├── mutation.py                  # 三色突变标记
│   ├── layout.py                    # UMAP-3D 演化树布局
│   └── report.py                    # 报告生成
├── db/                              # 数据库操作
│   ├── __init__.py
│   └── sqlite_ops.py               # SQLite CRUD + Schema 管理
└── run.py                           # 主入口：端到端流水线编排
```

### 8.1 与 V14-B 的关系

| 维度 | 说明 |
|------|------|
| 算法吸收 | V14-B 的 SPC Main Path、VGAE、Limitation Tracking、KeystoneScore、三色突变等核心算法在新架构中重新实现 |
| 架构重塑 | 不保留 V14-B 的 `echelon/v14b/` 目录结构和 9 步命名，按功能领域重新组织 |
| 数据独立 | 从 OpenAlex API 独立爬取，不读取 V14-B 的 SQLite 数据库 |
| 参考蓝本 | V14-B 代码（`/Users/lihaisen/Downloads/echelon_mvp0a 3/`）作为实现参考，理解算法细节后在新代码中重写 |

---

## 10. 文献支撑矩阵（仅含已验证的 A*/A 级论文）

### 10.1 核心文献（直接支撑设计决策）

| 设计决策 | 论文 | 期刊/会议 | 机构 | 级别 |
|---------|------|----------|------|------|
| SEMNET 方法论 | Krenn & Zeilinger, "Predicting research trends with semantic and neural networks" | PNAS 2020, 被引 200+ | Max Planck Institute | A* |
| SEMNET 大规模验证 | Gu & Krenn, "Forecasting high-impact research topics via ML on evolving knowledge graphs" | ML: Science and Technology 2025, AUC=0.948 | Max Planck Institute | A |
| LLM 概念提取 | Marwitz et al., "Predicting new research directions using LLMs and concept graphs" | Nature Machine Intelligence 2026 | KIT (Karlsruhe) | A* |
| LLM 科学信息提取 | Dagdelen et al., "Structured information extraction from scientific text with LLMs" | Nature Communications 2024 | Lawrence Berkeley Lab, UC Berkeley | A* |
| GBDT 链接预测 | Lu, "Predicting research trends with GBDT and time-aware GNN" | IEEE BigData 2022, Science4Cast 冠军 | — | A |
| 时序图基准 | Yi et al., "TGB-Seq: Challenging temporal GNNs with complex sequential dynamics" | ICLR 2025 | 多机构 | A* |
| Meta-learning 融合 | Ghasemian et al., "Stacking models for nearly optimal link prediction" | PNAS 2020, 203 算法 × 550 网络 | Colorado Boulder, Harvard, USC, Santa Fe Institute | A* |
| KGE+共现混合 | Nguyen et al., "NoGE: Node co-occurrence based GNNs for KG link prediction" | WSDM 2022 | Oracle Labs, Monash, VinAI | A |
| KGE 预测多样性 | Zhu et al., "Predictive multiplicity of KGE in link prediction" | EMNLP 2024 Findings | Stuttgart, Bosch AI, Cardiff | A |
| 增量 KGE | Liu et al., "FastKGE: Fast and continual KG embedding via incremental LoRA" | IJCAI 2024 | Southeast University, 中科院计算所 | A* |
| VGAE 升级 | Cho, "Decoupled variational graph autoencoder for link prediction" (D-VGAE) | WWW 2024 | Chung-Ang University | A* |

### 10.2 理论基础文献

| 维度 | 论文 | 期刊/会议 | 级别 |
|------|------|----------|------|
| 多层网络理论 | Kivela et al., "Multilayer networks" | J. Complex Networks 2014, 被引 4000+ | A* |
| 异构图注意力 | Wang et al., "Heterogeneous graph attention network (HAN)" | WWW 2019, 被引 2500+ | A* |
| 科学计量学 | Chen, "CiteSpace II: Detecting emerging trends" | JASIST 2006, 被引 7000+ | A* |
| 科学的科学 | Fortunato et al., "Science of science" | Science 2018, 被引 990+ | A* |
| 新兴技术定义 | Rotolo, Hicks & Martin, "What is an emerging technology?" | Research Policy 2015 | A |

### 10.3 被排除的论文（权威性审计未通过）

| 论文 | 声称期刊 | 实际状态 | 排除原因 |
|------|---------|---------|---------|
| Song et al. — LLM 关键词提取 | COLING 2025 | 仅 arXiv 预印本 | 期刊归属虚假 |
| Dobbins — 分层归一化 | Research Synthesis Methods 2025 | 仅 arXiv 预印本 | 期刊归属虚假 |

---

## 9. 技术依赖

### 9.1 全部依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| pydantic | ≥2.0 | 数据模型验证（Paper schema） |
| sentence-transformers | ≥2.0 | all-mpnet-base-v2 嵌入（论文+概念） |
| keybert | ≥0.8 | 概念提取校验通道 |
| torch | ≥2.0 | VGAE + KGE + NN 训练 |
| torch-geometric | ≥2.4 | VGAE 图神经网络 |
| xgboost | ≥2.0 | GBDT 链接预测 |
| lightgbm | ≥4.0 | GBDT 链接预测集成 |
| networkx | ≥3.0 | 引文图 + 概念共现图构建 + 拓扑特征 |
| scikit-learn | ≥1.3 | 层次聚类 + RF 元学习器 + TF-IDF/SVD |
| transformers | ≥4.30 | SciBERT 引用分类 |
| spacy | ≥3.5 | lemmatization 前处理 |
| shap | ≥0.43 | 特征重要性可解释性 |
| umap-learn | ≥0.5 | UMAP-3D 演化树布局 |
| requests | ≥2.28 | OpenAlex API 调用 |
| ulid-py | ≥1.1 | ULID 主键生成 |
| tiktoken | ≥0.5 | LLM token 计数 |

---

## 11. 分阶段交付计划

### Phase 1：数据采集 + 概念提取 + 共现网络（第 1 周）

- [ ] S0 OpenAlex API 爬取 200 篇验证数据集
- [ ] S1 数据入库 + Pydantic 验证 + 嵌入计算
- [ ] S5d 概念提取流水线（LLM + KeyBERT 双通道）
- [ ] 同义词归一化（嵌入聚类）
- [ ] 概念共现网络构建（年份切片）
- [ ] 200 篇验证数据集端到端跑通

**验证标准**：paper_identity 表 200 篇入库，concept_nodes 表 ≥ 100 个概念，paper_concepts 表覆盖率 ≥ 90%

### Phase 2：论文级预测 + 概念级预测 + 融合（第 2 周）

- [ ] S2-S4 论文级图谱层（SPC Main Path、KeystoneScore、子图构建）
- [ ] S5a-S5c 论文级预测层（SciBERT 引用分类、VGAE、Limitation）
- [ ] S5e 概念对特征工程（30+ 特征）+ KGE + GBDT/NN 链接预测
- [ ] S6 四路 Meta-learning Stacking 融合
- [ ] 200 篇端到端跑通

**验证标准**：概念级 AUC ≥ 0.85，四路融合输出 future_directions 表

### Phase 3（后续）：规模化 + 模型升级

- [ ] 扩展到 20 万篇（光学 + AI）
- [ ] VGAE → D-VGAE 升级
- [ ] S7-S9 输出层（三色突变 + UMAP-3D + 报告）
- [ ] 性能优化（批处理、增量更新）
- [ ] K8s 部署到火山引擎

---

## 12. 非功能需求（不变）

| 需求 | 指标 |
|------|------|
| 概念提取速度 | ≥ 10 篇/秒（KeyBERT）；LLM 受 API 限速 |
| 链接预测训练 | ≤ 30 分钟（200 篇数据集） |
| 端到端流水线 | ≤ 2 小时（20 万篇，含概念提取） |
| 内存峰值 | ≤ 16 GB（200 篇）；≤ 64 GB（20 万篇） |
| 存储增量 | ≤ 500 MB（4 张新表，20 万篇数据） |

---

## 13. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| LLM 概念提取成本高 | 20 万篇 × API 调用 = 高额费用 | 先用 KeyBERT 批量提取，LLM 仅处理 KeyBERT 置信度低的论文（混合策略） |
| 概念数量爆炸 | 20 万篇可能产生 10 万+ 概念 | 设置最低出现频次阈值（≥ 3 篇），过滤长尾噪声概念 |
| GBDT 过拟合 | 验证集 AUC 高但泛化差 | 时序严格划分（未来数据绝对不泄漏），5-fold 交叉验证 |
| 共现网络稀疏 | 200 篇数据集概念对稀疏 | Phase 1 重点验证流水线正确性，AUC 指标在 20 万篇时才有意义 |
| 融合层增加复杂度 | Meta-learning 需要足够训练数据 | Phase 1 先用简单加权融合，数据量足够后切换到 Stacking |

---

## 14. 设计决策记录 (ADR)

### ADR-001：概念提取选择 LLM 主通道 + KeyBERT 校验

**决策**：LLM 做主通道概念提取，KeyBERT 做补充校验（双通道互补）  
**替代方案**：(A) 纯 KeyBERT (B) 纯 LLM (C) SciBERT NER  
**选择理由**：Nature MI 2026 端到端验证 LLM 概念提取 → 概念图谱路径；KeyBERT 捕获 LLM 可能遗漏的表面关键词（重合率低，互补性强）  
**权威性审计**：A* 级论文支撑（Nature MI, Nature Comms）

### ADR-002：共现作为特征输入 KGE，而非独立共现网络

**决策**：共现频率/时序/拓扑作为 KGE 的辅助特征，而非独立建图  
**替代方案**：(A) 纯共现网络（Kulkarni 原始方案）(B) 纯 KGE (C) 混合  
**选择理由**：NoGE (WSDM 2022) 证明共现+KGE 结合在基准数据集达 SOTA；纯共现丢失语义关系  
**权威性审计**：A 级论文支撑（WSDM, IJCAI, EMNLP）

### ADR-003：四路 Meta-learning Stacking 融合

**决策**：RF 元学习器自动学习最优融合权重  
**替代方案**：(A) 简单交集 (B) 人工加权 (C) Attention-based  
**选择理由**：Ghasemian (PNAS 2020) 在 203 算法 × 550 网络上证明无单一算法普适，Stacking 接近最优  
**权威性审计**：A* 级论文支撑（PNAS）

### ADR-004：同义词归一化选择纯嵌入聚类

**决策**：SciBERT 嵌入 + 余弦阈值 0.85 + 层次聚类  
**替代方案**：(A) 嵌入粗筛+LLM 精判分层 (B) 纯规则匹配  
**选择理由**：分层方案（嵌入+LLM）的文献支撑仅有预印本，权威性不足。纯嵌入聚类是成熟技术，Phase 1 足够有效  
**回退条件**：Phase 2 若归一化质量不达标，再引入 LLM 精判层

### ADR-005：VGAE 升级路径

**决策**：Phase 2 实现标准 VGAE（参考 V14-B 算法），Phase 3 升级到 D-VGAE  
**依据**：D-VGAE (WWW 2024, A*) 解耦 homophily/popularity，解决标准 VGAE 内积解码器根本缺陷  
**分阶段理由**：Phase 2 先跑通四路融合框架，Phase 3 再优化单路性能

### ADR-006：独立数据采集，不复用 V14-B 数据库

**决策**：从 OpenAlex API 独立爬取数据，不读取 V14-B 的 SQLite  
**替代方案**：直接读取 V14-B 的 `echelon_library.sqlite3` 或 `v14_pilot.sqlite3`  
**选择理由**：新项目需要完整掌控数据流水线，V14-B 数据格式可能与新 schema 不一致，且独立数据采集便于后续扩展到新领域和新数据源  
**风险**：初期多花 1-2 天搭建爬取流水线，但长期收益远大于短期成本
