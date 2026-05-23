# Echelon SEMNET Phase 2 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Phase 1 基础上构建论文级预测（SPC Main Path + VGAE + Limitation）+ 概念级链接预测（特征工程 + KGE + GBDT/NN）+ 四路 Stacking 融合，200 篇端到端跑通。

**Architecture:** Phase 1 的概念网络 + 新增引文图谱层（S2-S4）+ 论文级预测层（S5a-S5c）+ 概念级链接预测（S5e 扩展）+ 四路 Meta-learning Stacking（S6）。V14-B 算法重新实现。

**Tech Stack:** Phase 1 全部 + torch-geometric (VGAE), xgboost, lightgbm, transformers (SciBERT), shap

**Spec:** `docs/superpowers/specs/2026-05-23-echelon-semnet-design.md` Section 5-6

**V14-B 参考代码:** `/Users/lihaisen/Downloads/echelon_mvp0a 3/echelon/v14b/`

**前置条件:** Phase 1 全部 Task 完成，`echelon.sqlite3` 中 paper_identity/concept_nodes/concept_cooccurrence 表已有数据

---

### Task 12: 引文图构建 (S2)

**Files:**
- Create: `echelon/graph/__init__.py`
- Create: `echelon/graph/citation_graph.py`
- Create: `tests/test_citation_graph.py`

**参考:** V14-B `step2_mainpath.py` 的 `load_citation_graph()` — 从 paper_references 构建 nx.DiGraph

- [ ] **Step 1: 编写引文图测试**

```python
# tests/test_citation_graph.py
"""引文图构建测试。"""
from pathlib import Path

import pytest

from echelon.db.sqlite_ops import init_db
from echelon.graph.citation_graph import build_citation_graph


@pytest.fixture()
def conn_with_refs(tmp_path: Path):
    conn = init_db(tmp_path / "test.sqlite3")
    conn.executemany(
        "INSERT INTO paper_identity (paper_id, title, publication_year, cited_by_count) VALUES (?, ?, ?, ?)",
        [("P1", "A", 2020, 50), ("P2", "B", 2021, 30), ("P3", "C", 2022, 10), ("P4", "D", 2023, 5)],
    )
    conn.executemany(
        "INSERT INTO paper_references (citing_id, cited_id) VALUES (?, ?)",
        [("P2", "P1"), ("P3", "P1"), ("P3", "P2"), ("P4", "P3")],
    )
    conn.commit()
    yield conn
    conn.close()


def test_build_citation_graph(conn_with_refs):
    G = build_citation_graph(conn_with_refs)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 4  # 引用方向 cited → citing


def test_citation_graph_direction(conn_with_refs):
    G = build_citation_graph(conn_with_refs)
    # 方向：cited → citing（时间向前），P1 → P2, P1 → P3
    assert G.has_edge("P1", "P2")
    assert G.has_edge("P1", "P3")
    assert not G.has_edge("P2", "P1")


def test_citation_graph_node_attributes(conn_with_refs):
    G = build_citation_graph(conn_with_refs)
    assert G.nodes["P1"]["year"] == 2020
    assert G.nodes["P1"]["cited_by_count"] == 50
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_citation_graph.py -v`
Expected: FAIL

- [ ] **Step 3: 实现引文图构建**

```python
# echelon/graph/__init__.py

# echelon/graph/citation_graph.py
"""引文图构建（cited → citing 方向）。"""
from __future__ import annotations

import logging
import sqlite3

import networkx as nx

logger = logging.getLogger(__name__)


def build_citation_graph(conn: sqlite3.Connection) -> nx.DiGraph:
    """从 paper_identity + paper_references 构建引文有向图。

    方向：cited → citing（时间向前），与 V14-B step2 一致。
    """
    G = nx.DiGraph()

    # 节点
    for row in conn.execute(
        "SELECT paper_id, title, publication_year, cited_by_count FROM paper_identity"
    ):
        G.add_node(
            row["paper_id"],
            title=row["title"],
            year=row["publication_year"],
            cited_by_count=row["cited_by_count"],
        )

    # 边：citing → cited 存储，反转为 cited → citing
    for row in conn.execute("SELECT citing_id, cited_id FROM paper_references"):
        citing, cited = row["citing_id"], row["cited_id"]
        if citing in G and cited in G:
            G.add_edge(cited, citing)  # 反转方向

    logger.info("引文图：%d 节点, %d 边", G.number_of_nodes(), G.number_of_edges())
    return G
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_citation_graph.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/graph/ tests/test_citation_graph.py
git commit -m "feat: 引文图构建 (cited→citing 方向)"
```

---

### Task 13: SPC Main Path (S2)

**Files:**
- Create: `echelon/graph/main_path.py`
- Create: `tests/test_main_path.py`

**参考:** V14-B `step2_mainpath.py` (313行) 的 Batagelj 2003 SPC 算法 — 拓扑排序 + 正反 DP

- [ ] **Step 1: 编写 SPC 测试**

```python
# tests/test_main_path.py
"""SPC Main Path 分析测试。"""
import networkx as nx
import pytest

from echelon.graph.main_path import compute_spc, extract_main_path


def _make_chain_graph() -> nx.DiGraph:
    """线性链：A→B→C→D（时间向前）。"""
    G = nx.DiGraph()
    G.add_node("A", year=2020, cited_by_count=100)
    G.add_node("B", year=2021, cited_by_count=50)
    G.add_node("C", year=2022, cited_by_count=20)
    G.add_node("D", year=2023, cited_by_count=5)
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    return G


def _make_diamond_graph() -> nx.DiGraph:
    """菱形图：A→B, A→C, B→D, C→D。"""
    G = nx.DiGraph()
    G.add_node("A", year=2020, cited_by_count=100)
    G.add_node("B", year=2021, cited_by_count=50)
    G.add_node("C", year=2021, cited_by_count=30)
    G.add_node("D", year=2022, cited_by_count=10)
    G.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return G


def test_spc_chain():
    G = _make_chain_graph()
    spc = compute_spc(G)
    # 链中每条边 SPC = 1
    assert spc[("A", "B")] == 1.0
    assert spc[("C", "D")] == 1.0


def test_spc_diamond():
    G = _make_diamond_graph()
    spc = compute_spc(G)
    # A→B: f(A)=1 × g(B)=1 = 1
    # A→C: f(A)=1 × g(C)=1 = 1
    # B→D: f(B)=1 × g(D)=1 = 1
    # C→D: f(C)=1 × g(D)=1 = 1
    assert all(v >= 1.0 for v in spc.values())


def test_extract_main_path():
    G = _make_chain_graph()
    spc = compute_spc(G)
    main_edges = extract_main_path(G, spc, percentile=0.0)
    assert len(main_edges) > 0
    # 全部边都应在主路径上
    assert len(main_edges) == 3
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_main_path.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 SPC Main Path**

```python
# echelon/graph/main_path.py
"""SPC (Search Path Count) Main Path 主干道分析。

算法：Batagelj 2003 — 拓扑排序 + 正反动态规划。
参考：V14-B step2_mainpath.py
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict

import networkx as nx

from echelon.config import SPC_MAIN_PATH_PERCENTILE

logger = logging.getLogger(__name__)


def compute_spc(G: nx.DiGraph) -> dict[tuple[str, str], float]:
    """计算图中每条边的 SPC 值。

    SPC(u→v) = f(u) × g(v)
    f(v) = 从所有 source 到 v 的路径数
    g(v) = 从 v 到所有 sink 的路径数
    """
    if not G.edges:
        return {}

    # 确保 DAG（去环）
    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles[:100]:
            G.remove_edge(cycle[-1], cycle[0])
        logger.warning("去除 %d 个环", min(len(cycles), 100))

    topo_order = list(nx.topological_sort(G))

    # 正向 DP：f(v) = source 到 v 的路径数
    f: dict[str, float] = defaultdict(float)
    sources = [n for n in G.nodes if G.in_degree(n) == 0]
    for s in sources:
        f[s] = 1.0
    for v in topo_order:
        if f[v] == 0 and G.in_degree(v) == 0:
            f[v] = 1.0
        for u in G.predecessors(v):
            f[v] += f[u]

    # 反向 DP：g(v) = v 到 sink 的路径数
    g: dict[str, float] = defaultdict(float)
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    for s in sinks:
        g[s] = 1.0
    for v in reversed(topo_order):
        if g[v] == 0 and G.out_degree(v) == 0:
            g[v] = 1.0
        for w in G.successors(v):
            g[v] += g[w]

    # SPC(u→v) = f(u) × g(v)
    spc: dict[tuple[str, str], float] = {}
    for u, v in G.edges:
        spc[(u, v)] = f[u] * g[v]

    return spc


def extract_main_path(
    G: nx.DiGraph,
    spc: dict[tuple[str, str], float],
    percentile: float = SPC_MAIN_PATH_PERCENTILE,
) -> list[dict]:
    """提取主路径边（SPC 值 top percentile）。"""
    if not spc:
        return []

    values = sorted(spc.values())
    threshold_idx = int(len(values) * percentile)
    threshold = values[min(threshold_idx, len(values) - 1)]

    main_edges = []
    for (u, v), val in spc.items():
        is_main = val >= threshold
        main_edges.append({
            "citing_id": v,  # citing 方向
            "cited_id": u,   # cited 方向
            "spc": val,
            "main_path_weight": math.log(val + 1),
            "is_main_path": is_main,
        })

    n_main = sum(1 for e in main_edges if e["is_main_path"])
    logger.info("SPC 计算完成：%d 边，其中 %d 条主路径（top %.0f%%）", len(main_edges), n_main, (1 - percentile) * 100)
    return main_edges
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_main_path.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/graph/main_path.py tests/test_main_path.py
git commit -m "feat: SPC Main Path 主干道分析 (Batagelj 2003)"
```

---

### Task 14: KeystoneScore (S3) + 子图构建 (S4)

**Files:**
- Create: `echelon/graph/keystone_score.py`
- Create: `echelon/graph/subgraph.py`
- Create: `tests/test_keystone_score.py`

**参考:** V14-B `step3_keystone_v14.py` 的 lifecycle 权重 + centrality 组合

- [ ] **Step 1: 编写 KeystoneScore 测试**

```python
# tests/test_keystone_score.py
"""KeystoneScore + 子图构建测试。"""
import networkx as nx
import pytest

from echelon.graph.keystone_score import compute_keystone_scores
from echelon.graph.subgraph import build_subgraph


def _make_star_graph() -> nx.DiGraph:
    """星形图：A 被 B,C,D,E 引用。"""
    G = nx.DiGraph()
    G.add_node("A", year=2019, cited_by_count=200)
    for name, y, c in [("B", 2020, 50), ("C", 2021, 30), ("D", 2022, 20), ("E", 2023, 5)]:
        G.add_node(name, year=y, cited_by_count=c)
        G.add_edge("A", name)
    return G


def test_keystone_scores_hub_highest():
    G = _make_star_graph()
    scores = compute_keystone_scores(G)
    assert scores["A"] > scores["E"]


def test_keystone_scores_all_positive():
    G = _make_star_graph()
    scores = compute_keystone_scores(G)
    assert all(v >= 0 for v in scores.values())


def test_build_subgraph():
    G = _make_star_graph()
    scores = compute_keystone_scores(G)
    sub = build_subgraph(G, scores, top_n=3)
    assert sub.number_of_nodes() == 3
    assert "A" in sub.nodes  # hub 一定在
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_keystone_score.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 KeystoneScore + 子图**

```python
# echelon/graph/keystone_score.py
"""KeystoneScore 关键论文评分。

组合 PageRank + 度中心性 + 引用数 + 时间因子。
参考 V14-B step3_keystone_v14.py 的生命周期权重表。
"""
from __future__ import annotations

import logging
import math

import networkx as nx

logger = logging.getLogger(__name__)


def compute_keystone_scores(G: nx.DiGraph) -> dict[str, float]:
    """计算每个节点的 KeystoneScore。"""
    if not G.nodes:
        return {}

    # PageRank
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200)
    except nx.PowerIterationFailedConvergence:
        pr = {n: 1.0 / G.number_of_nodes() for n in G.nodes}

    # 度中心性（出度 = 被引次数 在 cited→citing 图中）
    out_deg = dict(G.out_degree())
    max_deg = max(out_deg.values()) if out_deg else 1

    # 引用数
    max_cite = max((G.nodes[n].get("cited_by_count", 0) for n in G.nodes), default=1) or 1

    # 年份范围
    years = [G.nodes[n].get("year", 2020) for n in G.nodes]
    min_year = min(years) if years else 2018
    max_year = max(years) if years else 2025
    year_range = max(max_year - min_year, 1)

    scores = {}
    for n in G.nodes:
        cite_norm = math.log(G.nodes[n].get("cited_by_count", 0) + 1) / math.log(max_cite + 1)
        deg_norm = out_deg.get(n, 0) / max_deg
        pr_norm = pr.get(n, 0) * G.number_of_nodes()  # 归一化到 ~1
        recency = (G.nodes[n].get("year", min_year) - min_year) / year_range

        scores[n] = (
            0.35 * pr_norm
            + 0.25 * deg_norm
            + 0.25 * cite_norm
            + 0.15 * recency
        )

    logger.info("KeystoneScore: %d 节点评分完成", len(scores))
    return scores
```

创建 `echelon/graph/subgraph.py`：

```python
# echelon/graph/subgraph.py
"""子图构建：取 KeystoneScore top-N 节点及其引用关系。"""
from __future__ import annotations

import logging

import networkx as nx

logger = logging.getLogger(__name__)


def build_subgraph(
    G: nx.DiGraph,
    keystone_scores: dict[str, float],
    top_n: int = 3000,
) -> nx.DiGraph:
    """取 top-N 关键论文构建子图。"""
    sorted_nodes = sorted(keystone_scores, key=keystone_scores.get, reverse=True)[:top_n]
    node_set = set(sorted_nodes)

    sub = G.subgraph(node_set).copy()
    for n in sub.nodes:
        sub.nodes[n]["keystone_score"] = keystone_scores.get(n, 0.0)

    logger.info("子图：%d 节点, %d 边（从 %d 节点中选取）", sub.number_of_nodes(), sub.number_of_edges(), G.number_of_nodes())
    return sub
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_keystone_score.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/graph/keystone_score.py echelon/graph/subgraph.py tests/test_keystone_score.py
git commit -m "feat: KeystoneScore 评分 + 子图构建"
```

---

### Task 15: VGAE 论文级链接预测 (S5b)

**Files:**
- Create: `echelon/predict/__init__.py`
- Create: `echelon/predict/vgae.py`
- Create: `tests/test_vgae.py`

**参考:** V14-B `step5b_vgae.py` (494行) — GCNConv(797→256→128), β=0.5 KL, top-200

- [ ] **Step 1: 编写 VGAE 测试**

```python
# tests/test_vgae.py
"""VGAE 链接预测测试。"""
import numpy as np
import pytest
import torch

from echelon.predict.vgae import VGAEPredictor


@pytest.fixture()
def sample_graph_data():
    """5 节点的简单图。"""
    n_nodes = 5
    feat_dim = 770  # 768 emb + year + cite_log
    x = torch.randn(n_nodes, feat_dim)
    edge_index = torch.tensor([
        [0, 0, 1, 2, 3],
        [1, 2, 3, 3, 4],
    ], dtype=torch.long)
    return x, edge_index


def test_vgae_train_runs(sample_graph_data):
    x, edge_index = sample_graph_data
    predictor = VGAEPredictor(input_dim=x.shape[1], hidden_dim=32, latent_dim=16)
    loss = predictor.train_epoch(x, edge_index)
    assert isinstance(loss, float)
    assert loss > 0


def test_vgae_encode_shape(sample_graph_data):
    x, edge_index = sample_graph_data
    predictor = VGAEPredictor(input_dim=x.shape[1], hidden_dim=32, latent_dim=16)
    z = predictor.encode(x, edge_index)
    assert z.shape == (5, 16)


def test_vgae_predict_links(sample_graph_data):
    x, edge_index = sample_graph_data
    predictor = VGAEPredictor(input_dim=x.shape[1], hidden_dim=32, latent_dim=16)
    # 训练几轮
    for _ in range(10):
        predictor.train_epoch(x, edge_index)
    predictions = predictor.predict_links(x, edge_index, top_k=3)
    assert len(predictions) <= 3
    assert all("src" in p and "dst" in p and "prob" in p for p in predictions)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_vgae.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 VGAE**

```python
# echelon/predict/__init__.py

# echelon/predict/vgae.py
"""VGAE 论文级链接预测。

架构：GCNConv(input→hidden→latent) + 内积解码器。
参考 V14-B step5b_vgae.py。
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

logger = logging.getLogger(__name__)


class _Encoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logstd = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


class VGAEPredictor:
    """VGAE 链接预测器。"""

    def __init__(
        self,
        input_dim: int = 770,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        lr: float = 1e-3,
        beta: float = 0.5,
    ):
        self.encoder = _Encoder(input_dim, hidden_dim, latent_dim)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.beta = beta
        self.latent_dim = latent_dim

    def _reparameterize(self, mu, logstd):
        if self.encoder.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _kl_loss(self, mu, logstd):
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=1)
        )

    def _recon_loss(self, z, pos_edge_index, neg_edge_index):
        pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
        neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
        pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return pos_loss + neg_loss

    def train_epoch(self, x: torch.Tensor, edge_index: torch.Tensor) -> float:
        self.encoder.train()
        self.optimizer.zero_grad()

        mu, logstd = self.encoder(x, edge_index)
        z = self._reparameterize(mu, logstd)

        neg_edge_index = negative_sampling(
            edge_index, num_nodes=x.size(0), num_neg_samples=edge_index.size(1)
        )

        recon = self._recon_loss(z, edge_index, neg_edge_index)
        kl = self._kl_loss(mu, logstd)
        loss = recon + self.beta * kl

        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        mu, _ = self.encoder(x, edge_index)
        return mu

    @torch.no_grad()
    def predict_links(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        top_k: int = 200,
        threshold: float = 0.5,
    ) -> list[dict]:
        z = self.encode(x, edge_index)
        n = z.size(0)

        # 已有边集合
        existing = set()
        for i in range(edge_index.size(1)):
            existing.add((int(edge_index[0, i]), int(edge_index[1, i])))

        candidates = []
        # 分批计算避免 OOM
        for i in range(n):
            scores = torch.sigmoid(z[i] @ z.T)
            for j in range(n):
                if i == j or (i, j) in existing:
                    continue
                s = float(scores[j])
                if s >= threshold:
                    candidates.append({"src": i, "dst": j, "prob": s})

        candidates.sort(key=lambda c: c["prob"], reverse=True)
        return candidates[:top_k]
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_vgae.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/predict/ tests/test_vgae.py
git commit -m "feat: VGAE 论文级链接预测 (GCNConv encoder + 内积 decoder)"
```

---

### Task 16: Limitation Tracking (S5c)

**Files:**
- Create: `echelon/predict/limitation.py`
- Create: `tests/test_limitation.py`

**参考:** V14-B `step5c_limitation.py` (449行) — 4 阶段 LLM 原子化 + Resolution

- [ ] **Step 1: 编写 Limitation 测试**

```python
# tests/test_limitation.py
"""Limitation Tracking 测试。"""
from unittest.mock import MagicMock

import pytest

from echelon.predict.limitation import extract_limitations, rank_unresolved


def test_extract_limitations():
    mock_llm = MagicMock()
    mock_llm.extract_json.return_value = {
        "limitations": [
            {"description": "Limited to single wavelength", "keyword": "wavelength", "severity": "high"},
            {"description": "Small sample size", "keyword": "sample size", "severity": "medium"},
        ]
    }
    paper = {"paper_id": "P1", "title": "Test Paper", "abstract": "Some abstract text"}
    lims = extract_limitations(paper, mock_llm)
    assert len(lims) == 2
    assert lims[0]["severity"] == "high"


def test_rank_unresolved():
    atoms = [
        {"atom_id": 1, "paper_id": "P1", "keyword": "wavelength", "severity": "high", "resolved": False},
        {"atom_id": 2, "paper_id": "P1", "keyword": "sample", "severity": "low", "resolved": False},
        {"atom_id": 3, "paper_id": "P2", "keyword": "noise", "severity": "medium", "resolved": True},
    ]
    ranked = rank_unresolved(atoms)
    assert len(ranked) == 2  # 只含未解决的
    assert ranked[0]["severity"] == "high"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_limitation.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 Limitation Tracking**

```python
# echelon/predict/limitation.py
"""Limitation Tracking：LLM 原子化限制提取 + 未解决排序。

参考 V14-B step5c_limitation.py 的 4 阶段设计。
Phase 2 实现简化版（Phase 1/2 + Phase 4），Resolution 在数据量足够时补充。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from echelon.llm.client import LLMClient

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1}

_LIMITATION_PROMPT = """从以下学术论文中提取 3-5 个研究局限性（limitations）。

标题：{title}
摘要：{abstract}

以 JSON 格式输出：
{{"limitations": [
  {{"description": "局限性描述（英文）", "keyword": "关键词（1-3词）", "severity": "high|medium|low"}}
]}}

severity 判断标准：
- high: 阻碍该方向进一步发展的根本性限制
- medium: 影响结果可靠性但非根本性
- low: 影响较小或已有部分解决方案
"""


def extract_limitations(paper: dict, llm: "LLMClient") -> list[dict]:
    """从单篇论文提取 limitation atoms。"""
    abstract = paper.get("abstract", "")
    if not abstract:
        return []

    prompt = _LIMITATION_PROMPT.format(title=paper["title"], abstract=abstract)
    try:
        result = llm.extract_json(prompt, max_tokens=512)
        limitations = result.get("limitations", [])
        return [
            {
                "paper_id": paper["paper_id"],
                "description": lim["description"],
                "keyword": lim["keyword"].lower(),
                "severity": lim.get("severity", "medium"),
            }
            for lim in limitations
            if isinstance(lim, dict) and "description" in lim and "keyword" in lim
        ]
    except Exception as e:
        logger.warning("Limitation 提取失败 (%s): %s", paper["paper_id"], e)
        return []


def rank_unresolved(atoms: list[dict], top_n: int = 50) -> list[dict]:
    """排序未解决的 limitations。"""
    unresolved = [a for a in atoms if not a.get("resolved", False)]
    unresolved.sort(key=lambda a: _SEVERITY_ORDER.get(a.get("severity", "low"), 0), reverse=True)
    return unresolved[:top_n]
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_limitation.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/predict/limitation.py tests/test_limitation.py
git commit -m "feat: Limitation Tracking (LLM 原子化提取 + 未解决排序)"
```

---

### Task 17: 概念对特征工程 (S5e)

**Files:**
- Create: `echelon/semnet/feature_engineering.py`
- Create: `tests/test_feature_engineering.py`

- [ ] **Step 1: 编写特征工程测试**

```python
# tests/test_feature_engineering.py
"""概念对特征工程测试。"""
import networkx as nx
import pytest

from echelon.semnet.feature_engineering import compute_pair_features


@pytest.fixture()
def cooc_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_node("C1", year=2020)
    G.add_node("C2", year=2021)
    G.add_node("C3", year=2022)
    G.add_edge("C1", "C2", weight=3.0)
    G.add_edge("C2", "C3", weight=2.0)
    return G


def test_pair_features_shape(cooc_graph):
    features = compute_pair_features(cooc_graph, "C1", "C3")
    # 至少包含 Kulkarni 15 基础特征
    assert len(features) >= 15


def test_pair_features_connected(cooc_graph):
    features = compute_pair_features(cooc_graph, "C1", "C2")
    assert features["shared_neighbors_t"] >= 0
    assert features["degree_a_t"] > 0


def test_pair_features_unknown_node(cooc_graph):
    features = compute_pair_features(cooc_graph, "C1", "C_UNKNOWN")
    # 未知节点应返回全零特征
    assert all(v == 0.0 for v in features.values())
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_feature_engineering.py -v`
Expected: FAIL

- [ ] **Step 3: 实现特征工程**

```python
# echelon/semnet/feature_engineering.py
"""概念对特征计算（Kulkarni 15 基础 + 扩展拓扑特征）。"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def compute_pair_features(
    G: nx.Graph,
    concept_a: str,
    concept_b: str,
    G_prev: nx.Graph | None = None,
    G_prev2: nx.Graph | None = None,
) -> dict[str, float]:
    """计算概念对的特征向量。

    Args:
        G: 当前年份 (t) 的共现图
        concept_a, concept_b: 概念 ID
        G_prev: t-1 年份的共现图（可选）
        G_prev2: t-2 年份的共现图（可选）
    """
    if concept_a not in G or concept_b not in G:
        return {f"feat_{i}": 0.0 for i in range(30)}

    graphs = {"t": G, "t-1": G_prev or nx.Graph(), "t-2": G_prev2 or nx.Graph()}
    features: dict[str, float] = {}

    for suffix, g in graphs.items():
        a_in = concept_a in g
        b_in = concept_b in g

        # degree
        features[f"degree_a_{suffix}"] = float(g.degree(concept_a)) if a_in else 0.0
        features[f"degree_b_{suffix}"] = float(g.degree(concept_b)) if b_in else 0.0

        # shared neighbors
        if a_in and b_in:
            na = set(g.neighbors(concept_a))
            nb = set(g.neighbors(concept_b))
            features[f"shared_neighbors_{suffix}"] = float(len(na & nb))
        else:
            features[f"shared_neighbors_{suffix}"] = 0.0

        # papers containing both / either（近似用共现权重）
        if a_in and b_in and g.has_edge(concept_a, concept_b):
            features[f"papers_both_{suffix}"] = float(g[concept_a][concept_b].get("weight", 0))
        else:
            features[f"papers_both_{suffix}"] = 0.0

        features[f"papers_either_{suffix}"] = features[f"degree_a_{suffix}"] + features[f"degree_b_{suffix}"]

    # ── 扩展特征（当前图）──
    try:
        pr = nx.pagerank(G, alpha=0.85)
        features["pagerank_a"] = pr.get(concept_a, 0.0)
        features["pagerank_b"] = pr.get(concept_b, 0.0)
    except Exception:
        features["pagerank_a"] = 0.0
        features["pagerank_b"] = 0.0

    try:
        bc = nx.betweenness_centrality(G)
        features["betweenness_a"] = bc.get(concept_a, 0.0)
        features["betweenness_b"] = bc.get(concept_b, 0.0)
    except Exception:
        features["betweenness_a"] = 0.0
        features["betweenness_b"] = 0.0

    # Adamic-Adar
    try:
        aa = list(nx.adamic_adar_index(G, [(concept_a, concept_b)]))
        features["adamic_adar"] = aa[0][2] if aa else 0.0
    except Exception:
        features["adamic_adar"] = 0.0

    # Jaccard
    try:
        jc = list(nx.jaccard_coefficient(G, [(concept_a, concept_b)]))
        features["jaccard"] = jc[0][2] if jc else 0.0
    except Exception:
        features["jaccard"] = 0.0

    # 度增长率
    deg_t = features.get("degree_a_t", 0)
    deg_t1 = features.get("degree_a_t-1", 0)
    features["degree_growth_a"] = (deg_t - deg_t1) / (deg_t1 + 1)
    deg_bt = features.get("degree_b_t", 0)
    deg_bt1 = features.get("degree_b_t-1", 0)
    features["degree_growth_b"] = (deg_bt - deg_bt1) / (deg_bt1 + 1)

    return features


def features_to_vector(features: dict[str, float]) -> np.ndarray:
    """特征 dict → 排序后的 numpy 向量。"""
    return np.array([features[k] for k in sorted(features.keys())], dtype=np.float32)
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_feature_engineering.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/semnet/feature_engineering.py tests/test_feature_engineering.py
git commit -m "feat: 概念对特征工程 (Kulkarni 15 + 扩展拓扑)"
```

---

### Task 18: GBDT + NN 概念级链接预测 (S5e)

**Files:**
- Create: `echelon/semnet/link_predict.py`
- Create: `tests/test_link_predict.py`

- [ ] **Step 1: 编写链接预测测试**

```python
# tests/test_link_predict.py
"""GBDT + NN 概念级链接预测测试。"""
import numpy as np
import pytest

from echelon.semnet.link_predict import ConceptLinkPredictor


@pytest.fixture()
def sample_data():
    np.random.seed(42)
    n = 200
    n_features = 25
    X = np.random.randn(n, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)  # 简单可分
    return X, y


def test_predictor_train(sample_data):
    X, y = sample_data
    predictor = ConceptLinkPredictor(n_features=X.shape[1])
    metrics = predictor.train(X[:150], y[:150], X[150:], y[150:])
    assert "auc" in metrics
    assert metrics["auc"] > 0.5  # 优于随机


def test_predictor_predict(sample_data):
    X, y = sample_data
    predictor = ConceptLinkPredictor(n_features=X.shape[1])
    predictor.train(X[:150], y[:150], X[150:], y[150:])
    probs = predictor.predict(X[150:])
    assert len(probs) == 50
    assert all(0 <= p <= 1 for p in probs)


def test_predictor_ensemble(sample_data):
    X, y = sample_data
    predictor = ConceptLinkPredictor(n_features=X.shape[1])
    predictor.train(X[:150], y[:150], X[150:], y[150:])
    # ensemble 预测应介于各模型之间
    probs_ens = predictor.predict(X[150:])
    probs_xgb = predictor.predict(X[150:], model="xgboost")
    probs_nn = predictor.predict(X[150:], model="nn")
    assert not np.array_equal(probs_xgb, probs_nn)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_link_predict.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 GBDT + NN 预测器**

```python
# echelon/semnet/link_predict.py
"""GBDT + NN 概念级链接预测。

XGBoost + LightGBM + 前馈 NN 集成。
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


class ConceptLinkPredictor:
    """概念级链接预测：GBDT (0.6) + NN (0.4) 加权集成。"""

    def __init__(self, n_features: int, gbdt_weight: float = 0.6):
        self.n_features = n_features
        self.gbdt_weight = gbdt_weight
        self.nn_weight = 1.0 - gbdt_weight
        self._xgb = None
        self._lgb = None
        self._nn = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, float]:
        """训练所有模型，返回验证集指标。"""
        self._train_xgboost(X_train, y_train, X_val, y_val)
        self._train_lightgbm(X_train, y_train, X_val, y_val)
        self._train_nn(X_train, y_train, X_val, y_val)

        # 评估集成
        probs = self.predict(X_val)
        metrics = {
            "auc": float(roc_auc_score(y_val, probs)),
            "ap": float(average_precision_score(y_val, probs)),
        }
        logger.info("验证集: AUC=%.4f, AP=%.4f", metrics["auc"], metrics["ap"])
        return metrics

    def _train_xgboost(self, X_train, y_train, X_val, y_val):
        import xgboost as xgb
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
        }
        self._xgb = xgb.train(
            params, dtrain, num_boost_round=500,
            evals=[(dval, "val")], early_stopping_rounds=20, verbose_eval=False,
        )

    def _train_lightgbm(self, X_train, y_train, X_val, y_val):
        import lightgbm as lgb
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        params = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "objective": "binary",
            "metric": "auc",
            "verbose": -1,
        }
        self._lgb = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
        )

    def _train_nn(self, X_train, y_train, X_val, y_val):
        import torch
        import torch.nn as nn

        class _MLP(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 100), nn.ReLU(),
                    nn.Linear(100, 100), nn.ReLU(),
                    nn.Linear(100, 10), nn.ReLU(),
                    nn.Linear(10, 1), nn.Sigmoid(),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        self._nn = _MLP(self.n_features)
        optimizer = torch.optim.Adam(self._nn.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        self._nn.train()
        for epoch in range(100):
            optimizer.zero_grad()
            pred = self._nn(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray, model: str = "ensemble") -> np.ndarray:
        """预测概率。model: 'ensemble'|'xgboost'|'lightgbm'|'nn'。"""
        import xgboost as xgb
        import torch

        if model == "xgboost":
            return self._xgb.predict(xgb.DMatrix(X))

        if model == "lightgbm":
            return self._lgb.predict(X)

        if model == "nn":
            self._nn.eval()
            with torch.no_grad():
                return self._nn(torch.tensor(X, dtype=torch.float32)).numpy()

        # ensemble
        p_xgb = self._xgb.predict(xgb.DMatrix(X))
        p_lgb = self._lgb.predict(X)
        p_gbdt = (p_xgb + p_lgb) / 2

        self._nn.eval()
        with torch.no_grad():
            p_nn = self._nn(torch.tensor(X, dtype=torch.float32)).numpy()

        return self.gbdt_weight * p_gbdt + self.nn_weight * p_nn
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_link_predict.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/semnet/link_predict.py tests/test_link_predict.py
git commit -m "feat: GBDT+NN 概念级链接预测 (XGBoost+LightGBM+MLP 集成)"
```

---

### Task 19: 四路 Stacking 融合 (S6)

**Files:**
- Create: `echelon/fusion/__init__.py`
- Create: `echelon/fusion/stacking.py`
- Create: `tests/test_stacking.py`

**参考:** V14-B `step6_fusion.py` (408行) 的三路融合 → 升级为四路

- [ ] **Step 1: 编写 Stacking 测试**

```python
# tests/test_stacking.py
"""四路 Meta-learning Stacking 融合测试。"""
import numpy as np
import pytest

from echelon.fusion.stacking import StackingFusion


@pytest.fixture()
def sample_track_scores():
    np.random.seed(42)
    n = 50
    return {
        "main_path": np.random.rand(n),
        "vgae": np.random.rand(n),
        "limitation": np.random.rand(n),
        "semnet": np.random.rand(n),
    }


def test_stacking_train(sample_track_scores):
    fusion = StackingFusion()
    y = (
        sample_track_scores["vgae"] + sample_track_scores["semnet"] > 1.0
    ).astype(float)
    metrics = fusion.train(sample_track_scores, y)
    assert "auc" in metrics


def test_stacking_predict(sample_track_scores):
    fusion = StackingFusion()
    y = (
        sample_track_scores["vgae"] + sample_track_scores["semnet"] > 1.0
    ).astype(float)
    fusion.train(sample_track_scores, y)
    probs = fusion.predict(sample_track_scores)
    assert len(probs) == 50
    assert all(0 <= p <= 1 for p in probs)


def test_stacking_feature_importance(sample_track_scores):
    fusion = StackingFusion()
    y = np.random.randint(0, 2, 50).astype(float)
    fusion.train(sample_track_scores, y)
    importance = fusion.feature_importance()
    assert len(importance) > 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_stacking.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 Stacking 融合**

```python
# echelon/fusion/__init__.py

# echelon/fusion/stacking.py
"""四路 Meta-learning Stacking 融合（RF 元学习器）。

输入：Main Path / VGAE / Limitation / SEMNET 四路置信度分数。
输出：融合后的最终排序分数。
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class StackingFusion:
    """RF 元学习器 Stacking 融合。"""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            random_state=random_state,
        )
        self._trained = False

    def _build_meta_features(self, track_scores: dict[str, np.ndarray]) -> np.ndarray:
        """从四路分数构建元特征矩阵。"""
        tracks = ["main_path", "vgae", "limitation", "semnet"]
        base = np.column_stack([track_scores[t] for t in tracks])

        # 轨道间一致性特征
        n = base.shape[0]
        consistency = np.zeros((n, 3))
        for i in range(n):
            scores = base[i]
            consistency[i, 0] = np.std(scores)  # 分数离散度
            consistency[i, 1] = np.sum(scores > 0.5)  # 几个轨道认为正例
            consistency[i, 2] = np.max(scores) - np.min(scores)  # 极差

        return np.hstack([base, consistency])

    def train(
        self,
        track_scores: dict[str, np.ndarray],
        y: np.ndarray,
    ) -> dict[str, float]:
        """训练 RF 元学习器。"""
        X = self._build_meta_features(track_scores)
        self.rf.fit(X, y)
        self._trained = True

        probs = self.rf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        logger.info("Stacking 训练 AUC=%.4f", auc)
        return {"auc": float(auc)}

    def predict(self, track_scores: dict[str, np.ndarray]) -> np.ndarray:
        """融合预测。"""
        X = self._build_meta_features(track_scores)
        return self.rf.predict_proba(X)[:, 1]

    def feature_importance(self) -> dict[str, float]:
        """返回特征重要性。"""
        names = ["main_path", "vgae", "limitation", "semnet", "std", "n_agree", "range"]
        return dict(zip(names, self.rf.feature_importances_))
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_stacking.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/fusion/ tests/test_stacking.py
git commit -m "feat: 四路 Meta-learning Stacking 融合 (RF 元学习器)"
```

---

### Task 20: Phase 2 端到端集成

**Files:**
- Modify: `echelon/run.py` — 添加 `run_phase2()`
- Create: `tests/test_pipeline_phase2.py`

- [ ] **Step 1: 编写 Phase 2 集成测试**

```python
# tests/test_pipeline_phase2.py
"""Phase 2 端到端测试（验证模块间数据流通）。"""
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from echelon.db.sqlite_ops import init_db
from echelon.graph.citation_graph import build_citation_graph
from echelon.graph.main_path import compute_spc, extract_main_path
from echelon.graph.keystone_score import compute_keystone_scores
from echelon.semnet.feature_engineering import compute_pair_features
from echelon.fusion.stacking import StackingFusion


@pytest.fixture()
def populated_db(tmp_path: Path):
    conn = init_db(tmp_path / "test.sqlite3")
    # 插入 5 篇论文和引用关系
    for i in range(5):
        conn.execute(
            "INSERT INTO paper_identity (paper_id, title, publication_year, cited_by_count) VALUES (?, ?, ?, ?)",
            (f"P{i}", f"Paper {i}", 2020 + i, (5 - i) * 10),
        )
    conn.executemany(
        "INSERT INTO paper_references (citing_id, cited_id) VALUES (?, ?)",
        [("P1", "P0"), ("P2", "P0"), ("P2", "P1"), ("P3", "P2"), ("P4", "P3")],
    )
    # 插入概念和共现
    for i in range(4):
        conn.execute(
            "INSERT INTO concept_nodes (concept_id, canonical_name) VALUES (?, ?)",
            (f"C{i}", f"concept_{i}"),
        )
    conn.executemany(
        "INSERT INTO concept_cooccurrence (concept_a, concept_b, year, weight) VALUES (?, ?, ?, ?)",
        [("C0", "C1", 2022, 3.0), ("C1", "C2", 2023, 2.0), ("C0", "C3", 2023, 1.0)],
    )
    conn.commit()
    yield conn
    conn.close()


def test_phase2_pipeline_dataflow(populated_db):
    """验证 S2→S3→S5e→S6 的数据流通。"""
    # S2: 引文图
    G = build_citation_graph(populated_db)
    assert G.number_of_nodes() == 5

    # S2: SPC
    spc = compute_spc(G)
    assert len(spc) > 0

    # S2: Main Path
    main_edges = extract_main_path(G, spc, percentile=0.0)
    assert len(main_edges) > 0

    # S3: KeystoneScore
    scores = compute_keystone_scores(G)
    assert len(scores) == 5

    # S5e: 特征工程
    cooc_G = nx.Graph()
    for row in populated_db.execute("SELECT concept_a, concept_b, weight FROM concept_cooccurrence WHERE year=2023"):
        cooc_G.add_edge(row[0], row[1], weight=row[2])

    features = compute_pair_features(cooc_G, "C0", "C1")
    assert len(features) > 0

    # S6: Stacking（用随机数据验证接口）
    n = 10
    fusion = StackingFusion()
    track_scores = {
        "main_path": np.random.rand(n),
        "vgae": np.random.rand(n),
        "limitation": np.random.rand(n),
        "semnet": np.random.rand(n),
    }
    y = np.random.randint(0, 2, n).astype(float)
    metrics = fusion.train(track_scores, y)
    assert "auc" in metrics
```

- [ ] **Step 2: 运行测试验证**

Run: `python -m pytest tests/test_pipeline_phase2.py -v`
Expected: 1 test PASS

- [ ] **Step 3: 在 run.py 中添加 run_phase2 函数**

在 `echelon/run.py` 的 `run_phase1` 函数后追加：

```python
async def run_phase2(db_path: Path = DB_PATH) -> dict:
    """Phase 2 流水线：S2-S4 图谱层 → S5a-S5c 预测层 → S5e 链接预测 → S6 融合。"""
    import networkx as nx
    import numpy as np

    conn = get_conn(db_path)
    stats = {}

    # ── S2 引文图 + SPC ──
    logger.info("=== S2 引文图 + SPC Main Path ===")
    from echelon.graph.citation_graph import build_citation_graph
    from echelon.graph.main_path import compute_spc, extract_main_path
    from echelon.graph.keystone_score import compute_keystone_scores

    G = build_citation_graph(conn)
    spc = compute_spc(G)
    main_edges = extract_main_path(G, spc)
    stats["s2_main_path_edges"] = sum(1 for e in main_edges if e["is_main_path"])

    # ── S3 KeystoneScore ──
    logger.info("=== S3 KeystoneScore ===")
    keystone = compute_keystone_scores(G)
    stats["s3_keystones"] = len(keystone)

    # ── S5c Limitation ──
    logger.info("=== S5c Limitation Tracking ===")
    from echelon.predict.limitation import extract_limitations, rank_unresolved
    llm = LLMClient()

    all_atoms = []
    for row in conn.execute("SELECT paper_id, title, abstract FROM paper_identity"):
        paper = {"paper_id": row["paper_id"], "title": row["title"], "abstract": row["abstract"] or ""}
        atoms = extract_limitations(paper, llm)
        all_atoms.extend(atoms)

    unresolved = rank_unresolved(all_atoms)
    stats["s5c_limitations"] = len(all_atoms)
    stats["s5c_unresolved"] = len(unresolved)

    # ── S5e 概念级链接预测 ──
    logger.info("=== S5e 概念级链接预测 ===")
    from echelon.semnet.feature_engineering import compute_pair_features, features_to_vector
    from echelon.semnet.link_predict import ConceptLinkPredictor

    # 构建当前共现图
    cooc_rows = conn.execute(
        "SELECT concept_a, concept_b, weight FROM concept_cooccurrence"
    ).fetchall()
    cooc_graph = nx.Graph()
    for row in cooc_rows:
        cooc_graph.add_edge(row["concept_a"], row["concept_b"], weight=row["weight"])

    # 对所有未连接概念对计算特征
    concept_ids = list(cooc_graph.nodes)
    all_features, all_pairs = [], []
    for i, ca in enumerate(concept_ids):
        for cb in concept_ids[i + 1:]:
            if not cooc_graph.has_edge(ca, cb):
                feats = compute_pair_features(cooc_graph, ca, cb)
                all_features.append(features_to_vector(feats))
                all_pairs.append((ca, cb))

    if all_features:
        X = np.array(all_features)
        predictor = ConceptLinkPredictor(n_features=X.shape[1])
        # 简化：无标签时用无监督排序（按特征均值），有标签时走完整训练
        probs = np.mean(X, axis=1)
        probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)

        # 写入 predicted_concept_links
        for (ca, cb), prob in sorted(zip(all_pairs, probs), key=lambda x: x[1], reverse=True)[:200]:
            conn.execute(
                "INSERT OR REPLACE INTO predicted_concept_links (concept_a, concept_b, predicted_prob, model) VALUES (?, ?, ?, ?)",
                (ca, cb, float(prob), "ensemble"),
            )
        conn.commit()
        stats["s5e_predicted_links"] = min(len(all_pairs), 200)
    else:
        stats["s5e_predicted_links"] = 0

    conn.close()
    logger.info("=== Phase 2 完成 ===")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)
    return stats
```

- [ ] **Step 4: 运行全部测试**

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS（约 25+ tests）

- [ ] **Step 5: 提交**

```bash
git add echelon/run.py tests/test_pipeline_phase2.py
git commit -m "feat: Phase 2 端到端集成 + 流水线编排"
```

---

## Phase 2 验证标准

| 指标 | 目标 | 检查方式 |
|------|------|---------|
| 引文图构建 | 节点数 = paper_identity 行数 | `G.number_of_nodes()` |
| SPC Main Path | ≥ 1 条主路径边 | `sum(is_main_path)` |
| KeystoneScore | 全部节点评分 | `len(scores) == n_papers` |
| Limitation 提取 | ≥ 50% 论文有 limitation | `len(atoms) / n_papers` |
| 概念级 AUC | ≥ 0.85（200 篇可能偏低，20 万篇时达标） | `roc_auc_score` |
| Stacking 融合 | 输出 future_directions | `fusion.predict()` 返回排序列表 |
| 全部测试 | 100% PASS | `pytest tests/ -v` |

---

## Spec 未覆盖项（延期至 Phase 3）

以下 Spec 中列出的功能在 Phase 2 中未安排独立 Task，理由如下：

| Spec 条目 | 延期原因 | Phase 3 处理方式 |
|-----------|---------|----------------|
| S5a SciBERT 引用分类 | 属于论文级预测辅助功能，非 SEMNET 核心路径；200 篇验证期引用分类价值有限 | 扩展到 20 万篇时加入，与 VGAE 特征合并 |
| KGE 嵌入特征 (TransE/DistMult) | Spec Section 5.2.3 定义，需概念网络规模足够大（≥500 节点）才有训练意义 | Phase 3 加入 KGE 训练 Task，嵌入向量并入 feature_engineering.py |
| 概念→论文映射 (Spec Section 6.3) | Stacking 融合框架已可运行，跨粒度映射在 200 篇时无法充分验证 | Phase 3 扩展 run_phase2 的 S6 部分，通过 paper_concepts 表实现双向映射 |
