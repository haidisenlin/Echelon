# Echelon SEMNET Phase 1 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 S0 数据爬取 → S1 入库 → S5d 概念提取 → 概念共现网络的完整 Phase 1 流水线，在 200 篇光学论文上端到端跑通。

**Architecture:** 从 OpenAlex API 独立爬取论文 → Pydantic v2 验证入库 SQLite → LLM+KeyBERT 双通道概念提取 → 嵌入聚类同义词归一化 → 年份切片概念共现网络。V14-B 代码仅作参考蓝本，不导入。

**Tech Stack:** Python 3.12, Pydantic v2, SQLite (WAL), sentence-transformers (all-mpnet-base-v2), KeyBERT, spaCy, httpx, ulid-py

**Spec:** `docs/superpowers/specs/2026-05-23-echelon-semnet-design.md`

**V14-B 参考代码:** `/Users/lihaisen/Downloads/echelon_mvp0a 3/echelon/`（只读参考，不导入）

---

### Task 1: 项目脚手架 + 配置 + DB Schema

**Files:**
- Create: `echelon/__init__.py`
- Create: `echelon/config.py`
- Create: `echelon/db/__init__.py`
- Create: `echelon/db/sqlite_ops.py`
- Create: `tests/__init__.py`
- Create: `tests/test_db.py`
- Create: `pyproject.toml`

**参考:** V14-B `config.py` (283行) 的超参管理模式；`db_schema.py` (309行) 的 PRAGMA WAL + init 模式

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[project]
name = "echelon"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "sentence-transformers>=2.0",
    "keybert>=0.8",
    "torch>=2.0",
    "networkx>=3.0",
    "scikit-learn>=1.3",
    "spacy>=3.5",
    "httpx>=0.27",
    "ulid-py>=1.1",
    "tiktoken>=0.5",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]
phase2 = [
    "torch-geometric>=2.4",
    "xgboost>=2.0",
    "lightgbm>=4.0",
    "transformers>=4.30",
    "shap>=0.43",
    "umap-learn>=0.5",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: 创建 echelon/config.py**

```python
"""全局配置：数据库路径、模型超参、API 密钥。"""
from pathlib import Path
import os

# ── 路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "echelon.sqlite3"

# ── OpenAlex ──
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "")
OPENALEX_PER_PAGE = 200
OPENALEX_PILOT_TOPICS = [
    "T10245",  # Optical Coherence Tomography
    "T10621",  # Adaptive Optics
    "T11473",  # Photonic Crystal Fibers
    "T12806",  # Metasurface and Metamaterial
]
OPENALEX_PILOT_SINCE = "2018-01-01"
OPENALEX_PILOT_UNTIL = "2025-12-31"

# ── 嵌入 ──
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
EMBEDDING_BATCH_SIZE = 64

# ── LLM ──
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
LLM_MAX_CONCEPTS_PER_PAPER = 20

# ── KeyBERT ──
KEYBERT_TOP_N = 10
KEYBERT_DIVERSITY = 0.7

# ── 概念归一化 ──
NORMALIZE_COSINE_THRESHOLD = 0.15  # distance，即 similarity >= 0.85
CONCEPT_DEDUP_SIMILARITY = 0.90
CONCEPT_MIN_FREQUENCY = 1  # Phase 1 验证期不过滤，Phase 2 改为 3

# ── 共现网络 ──
COOCCURRENCE_TIME_DECAY = 0.8  # lambda

# ── SPC Main Path ──
SPC_MAIN_PATH_PERCENTILE = 0.99  # top 1% 边为主路径

# ── 调试 ──
LIMIT: int | None = None  # 设置后截断爬取数量
CONCURRENCY = 5
```

- [ ] **Step 3: 创建 echelon/db/sqlite_ops.py**

```python
"""SQLite CRUD + Schema 管理。"""
import sqlite3
from pathlib import Path

_DDL_BASE = """
CREATE TABLE IF NOT EXISTS paper_identity (
    paper_id         TEXT PRIMARY KEY,
    openalex_id      TEXT,
    title            TEXT NOT NULL,
    abstract         TEXT,
    publication_year  INTEGER,
    cited_by_count    INTEGER DEFAULT 0,
    source            TEXT,
    topics_json       TEXT,
    created_at        TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS paper_references (
    citing_id  TEXT NOT NULL,
    cited_id   TEXT NOT NULL,
    PRIMARY KEY (citing_id, cited_id)
);

CREATE TABLE IF NOT EXISTS paper_embeddings (
    paper_id   TEXT PRIMARY KEY,
    embedding  BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS concept_nodes (
    concept_id     TEXT PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    first_seen_year INTEGER,
    total_papers    INTEGER DEFAULT 0,
    embedding       BLOB
);

CREATE TABLE IF NOT EXISTS paper_concepts (
    paper_id    TEXT NOT NULL,
    concept_id  TEXT NOT NULL,
    score       REAL,
    method      TEXT,
    PRIMARY KEY (paper_id, concept_id)
);

CREATE TABLE IF NOT EXISTS concept_cooccurrence (
    concept_a  TEXT NOT NULL,
    concept_b  TEXT NOT NULL,
    year       INTEGER NOT NULL,
    weight     REAL,
    PRIMARY KEY (concept_a, concept_b, year)
);

CREATE TABLE IF NOT EXISTS predicted_concept_links (
    concept_a      TEXT NOT NULL,
    concept_b      TEXT NOT NULL,
    predicted_prob REAL,
    model          TEXT,
    features_json  TEXT,
    created_at     TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (concept_a, concept_b, model)
);

CREATE INDEX IF NOT EXISTS idx_paper_concepts_paper ON paper_concepts(paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_concepts_concept ON paper_concepts(concept_id);
CREATE INDEX IF NOT EXISTS idx_cooccurrence_year ON concept_cooccurrence(year);
CREATE INDEX IF NOT EXISTS idx_predicted_prob ON predicted_concept_links(predicted_prob DESC);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    """初始化数据库：PRAGMA 优化 + 建表。"""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL_BASE)
    return conn


def get_conn(db_path: Path) -> sqlite3.Connection:
    """获取连接，不存在则初始化。"""
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn
    return init_db(db_path)
```

- [ ] **Step 4: 创建 echelon/__init__.py 和 echelon/db/__init__.py**

```python
# echelon/__init__.py
"""Echelon — 企业级科研趋势预测引擎。"""

# echelon/db/__init__.py
```

- [ ] **Step 5: 编写 DB 初始化测试**

```python
# tests/test_db.py
"""数据库初始化和 schema 测试。"""
import sqlite3
from pathlib import Path

import pytest

from echelon.db.sqlite_ops import init_db, get_conn


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.sqlite3"


def test_init_db_creates_tables(db_path: Path):
    conn = init_db(db_path)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    expected = {
        "paper_identity",
        "paper_references",
        "paper_embeddings",
        "concept_nodes",
        "paper_concepts",
        "concept_cooccurrence",
        "predicted_concept_links",
    }
    assert expected.issubset(tables)
    conn.close()


def test_init_db_wal_mode(db_path: Path):
    conn = init_db(db_path)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    conn.close()


def test_get_conn_idempotent(db_path: Path):
    conn1 = init_db(db_path)
    conn1.close()
    conn2 = get_conn(db_path)
    count = conn2.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table'"
    ).fetchone()[0]
    assert count >= 7
    conn2.close()
```

- [ ] **Step 6: 运行测试验证**

Run: `cd /Users/lihaisen/PycharmProjects/Echelon && python -m pytest tests/test_db.py -v`
Expected: 3 tests PASS

- [ ] **Step 7: 提交**

```bash
git add pyproject.toml echelon/ tests/
git commit -m "feat: 项目脚手架 + 配置 + DB Schema 初始化"
```

---

### Task 2: Paper Schema (Pydantic v2)

**Files:**
- Create: `echelon/schema/__init__.py`
- Create: `echelon/schema/paper.py`
- Create: `tests/test_schema.py`

**参考:** V14-B `schema/paper.py` (137行) 的 Paper 模型 + ULID + validators

- [ ] **Step 1: 编写 Paper schema 测试**

```python
# tests/test_schema.py
"""Paper Pydantic schema 测试。"""
from datetime import date

import pytest

from echelon.schema.paper import Paper


def test_paper_ulid_auto_generated():
    p = Paper(title="Test Paper", publication_year=2024)
    assert len(p.paper_id) == 26  # ULID 长度


def test_paper_from_openalex_json():
    raw = {
        "id": "https://openalex.org/W123",
        "title": "Adaptive Optics for Microscopy",
        "abstract_inverted_index": {"Adaptive": [0], "optics": [1], "enables": [2]},
        "publication_date": "2023-05-15",
        "cited_by_count": 42,
        "primary_topic": {
            "id": "https://openalex.org/T10621",
            "display_name": "Adaptive Optics",
        },
        "referenced_works": [
            "https://openalex.org/W456",
            "https://openalex.org/W789",
        ],
    }
    p = Paper.from_openalex(raw)
    assert p.openalex_id == "W123"
    assert p.title == "Adaptive Optics for Microscopy"
    assert "Adaptive optics enables" in p.abstract
    assert p.publication_year == 2023
    assert p.cited_by_count == 42
    assert len(p.referenced_work_ids) == 2


def test_paper_abstract_from_inverted_index():
    inv = {"Hello": [0], "world": [1], "foo": [3], "bar": [2]}
    p = Paper(
        title="T",
        publication_year=2024,
        abstract_inverted_index=inv,
    )
    assert p.abstract == "Hello world bar foo"


def test_paper_rejects_empty_title():
    with pytest.raises(ValueError):
        Paper(title="", publication_year=2024)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'echelon.schema.paper'`

- [ ] **Step 3: 实现 Paper schema**

```python
# echelon/schema/__init__.py

# echelon/schema/paper.py
"""Pydantic v2 Paper 模型，ULID 主键。"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator
import ulid


def _ulid_new() -> str:
    return str(ulid.new())


def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    """OpenAlex abstract_inverted_index → 纯文本。"""
    if not inverted_index:
        return ""
    pairs: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort(key=lambda x: x[0])
    return " ".join(w for _, w in pairs)


class Paper(BaseModel):
    """论文数据模型。"""

    paper_id: str = Field(default_factory=_ulid_new)
    openalex_id: str | None = None
    title: str = Field(min_length=1)
    abstract: str | None = None
    abstract_inverted_index: dict[str, list[int]] | None = Field(
        default=None, exclude=True
    )
    publication_year: int
    cited_by_count: int = 0
    source: str | None = None
    topics_json: str | None = None
    referenced_work_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _build_abstract_from_inverted_index(self) -> Paper:
        if self.abstract is None and self.abstract_inverted_index:
            self.abstract = _reconstruct_abstract(self.abstract_inverted_index)
        return self

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        return v.strip()

    @classmethod
    def from_openalex(cls, raw: dict) -> Paper:
        """从 OpenAlex Work JSON 构造 Paper。"""
        oa_id = raw.get("id", "")
        if "openalex.org/" in oa_id:
            oa_id = oa_id.split("/")[-1]

        pub_date = raw.get("publication_date", "")
        year = int(pub_date[:4]) if pub_date else 0

        topic = raw.get("primary_topic") or {}
        topic_id = topic.get("id", "")
        if "openalex.org/" in str(topic_id):
            topic_id = topic_id.split("/")[-1]

        refs = raw.get("referenced_works", [])
        ref_ids = [r.split("/")[-1] if "/" in r else r for r in refs]

        source_obj = raw.get("primary_location", {}) or {}
        source_venue = (source_obj.get("source") or {}).get("display_name")

        import json

        return cls(
            openalex_id=oa_id,
            title=raw.get("title") or "Untitled",
            abstract_inverted_index=raw.get("abstract_inverted_index"),
            publication_year=year,
            cited_by_count=raw.get("cited_by_count", 0),
            source=source_venue,
            topics_json=json.dumps(
                {"id": topic_id, "name": topic.get("display_name", "")},
                ensure_ascii=False,
            )
            if topic_id
            else None,
            referenced_work_ids=ref_ids,
        )
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_schema.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/schema/ tests/test_schema.py
git commit -m "feat: Paper Pydantic v2 schema + OpenAlex JSON 解析"
```

---

### Task 3: OpenAlex API 爬取 (S0)

**Files:**
- Create: `echelon/ingest/__init__.py`
- Create: `echelon/ingest/fetcher.py`
- Create: `tests/test_fetcher.py`

**参考:** V14-B `openalex_client.py` (184行) 的 cursor pagination + polite pool 模式

- [ ] **Step 1: 编写 fetcher 测试（使用 mock）**

```python
# tests/test_fetcher.py
"""OpenAlex API fetcher 测试。"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from echelon.ingest.fetcher import fetch_works_by_topics


def _make_oa_response(results: list[dict], next_cursor: str | None) -> dict:
    return {
        "meta": {"count": len(results), "next_cursor": next_cursor},
        "results": results,
    }


def _make_work(work_id: str, title: str) -> dict:
    return {
        "id": f"https://openalex.org/{work_id}",
        "title": title,
        "abstract_inverted_index": {"test": [0]},
        "publication_date": "2023-01-01",
        "cited_by_count": 10,
        "primary_topic": {
            "id": "https://openalex.org/T10245",
            "display_name": "OCT",
        },
        "referenced_works": [],
    }


@pytest.mark.asyncio
async def test_fetch_single_page():
    page1 = _make_oa_response(
        [_make_work("W1", "Paper 1"), _make_work("W2", "Paper 2")],
        next_cursor=None,
    )

    mock_client = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.json.return_value = page1
    mock_resp.raise_for_status = lambda: None
    mock_client.get.return_value = mock_resp

    works = []
    async for w in fetch_works_by_topics(
        topic_ids=["T10245"],
        since="2023-01-01",
        until="2023-12-31",
        http_client=mock_client,
    ):
        works.append(w)

    assert len(works) == 2
    assert works[0]["title"] == "Paper 1"


@pytest.mark.asyncio
async def test_fetch_pagination():
    page1 = _make_oa_response(
        [_make_work("W1", "P1")], next_cursor="abc123"
    )
    page2 = _make_oa_response(
        [_make_work("W2", "P2")], next_cursor=None
    )

    mock_client = AsyncMock()
    mock_resp1, mock_resp2 = AsyncMock(), AsyncMock()
    mock_resp1.json.return_value = page1
    mock_resp1.raise_for_status = lambda: None
    mock_resp2.json.return_value = page2
    mock_resp2.raise_for_status = lambda: None
    mock_client.get.side_effect = [mock_resp1, mock_resp2]

    works = []
    async for w in fetch_works_by_topics(
        topic_ids=["T10245"],
        since="2023-01-01",
        until="2023-12-31",
        http_client=mock_client,
    ):
        works.append(w)

    assert len(works) == 2
    assert mock_client.get.call_count == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_fetcher.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 fetcher**

```python
# echelon/ingest/__init__.py

# echelon/ingest/fetcher.py
"""OpenAlex API 游标分页爬取。"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

from echelon.config import OPENALEX_MAILTO, OPENALEX_PER_PAGE

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.openalex.org/works"

_SELECT_FIELDS = ",".join([
    "id", "doi", "title", "abstract_inverted_index",
    "publication_date", "primary_topic", "authorships",
    "referenced_works", "cited_by_count", "language",
    "primary_location",
])


async def fetch_works_by_topics(
    *,
    topic_ids: list[str],
    since: str,
    until: str,
    per_page: int = OPENALEX_PER_PAGE,
    max_results: int | None = None,
    http_client: "httpx.AsyncClient | None" = None,
) -> AsyncIterator[dict]:
    """按 topic 列表爬取 OpenAlex Works，游标分页。"""
    import httpx as _httpx

    own_client = http_client is None
    if own_client:
        http_client = _httpx.AsyncClient(timeout=30.0)

    try:
        for topic_id in topic_ids:
            count = 0
            cursor = "*"
            topic_filter = (
                f"primary_topic.id:https://openalex.org/{topic_id}"
                if not topic_id.startswith("http") else
                f"primary_topic.id:{topic_id}"
            )

            while cursor:
                params = {
                    "filter": (
                        f"{topic_filter},"
                        f"from_publication_date:{since},"
                        f"to_publication_date:{until},"
                        "is_retracted:false,"
                        "has_abstract:true"
                    ),
                    "select": _SELECT_FIELDS,
                    "per_page": per_page,
                    "cursor": cursor,
                }
                if OPENALEX_MAILTO:
                    params["mailto"] = OPENALEX_MAILTO

                resp = await http_client.get(_BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

                for work in data.get("results", []):
                    yield work
                    count += 1
                    if max_results and count >= max_results:
                        return

                cursor = data.get("meta", {}).get("next_cursor")
                if cursor:
                    await asyncio.sleep(0.1)  # polite rate limit

            logger.info("Topic %s: fetched %d works", topic_id, count)
    finally:
        if own_client:
            await http_client.aclose()
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_fetcher.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/ingest/ tests/test_fetcher.py
git commit -m "feat: OpenAlex API 游标分页爬取"
```

---

### Task 4: 数据入库 (S1)

**Files:**
- Create: `echelon/ingest/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: 编写入库测试**

```python
# tests/test_ingest.py
"""数据入库测试。"""
from pathlib import Path

import pytest

from echelon.db.sqlite_ops import init_db
from echelon.ingest.ingest import ingest_papers
from echelon.schema.paper import Paper


@pytest.fixture()
def conn(tmp_path: Path):
    c = init_db(tmp_path / "test.sqlite3")
    yield c
    c.close()


def _sample_papers() -> list[Paper]:
    return [
        Paper(
            title="Paper A",
            openalex_id="W001",
            abstract="Adaptive optics for imaging",
            publication_year=2022,
            cited_by_count=10,
            referenced_work_ids=["W002"],
        ),
        Paper(
            title="Paper B",
            openalex_id="W002",
            abstract="Deep learning in microscopy",
            publication_year=2021,
            cited_by_count=25,
        ),
    ]


def test_ingest_papers_basic(conn):
    papers = _sample_papers()
    inserted = ingest_papers(conn, papers)
    assert inserted == 2

    rows = conn.execute("SELECT count(*) FROM paper_identity").fetchone()[0]
    assert rows == 2


def test_ingest_papers_references(conn):
    papers = _sample_papers()
    ingest_papers(conn, papers)

    refs = conn.execute("SELECT count(*) FROM paper_references").fetchone()[0]
    assert refs >= 0  # W002 可能不在库中，引用仍应被记录


def test_ingest_papers_idempotent(conn):
    papers = _sample_papers()
    ingest_papers(conn, papers)
    ingest_papers(conn, papers)  # 重复入库

    rows = conn.execute("SELECT count(*) FROM paper_identity").fetchone()[0]
    assert rows == 2  # 不应重复
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 ingest**

```python
# echelon/ingest/ingest.py
"""JSONL/JSON → SQLite 入库。"""
from __future__ import annotations

import json
import logging
import sqlite3

from echelon.schema.paper import Paper

logger = logging.getLogger(__name__)


def ingest_papers(conn: sqlite3.Connection, papers: list[Paper]) -> int:
    """批量入库论文到 paper_identity + paper_references。返回新插入数。"""
    inserted = 0
    # 先查出已有 openalex_id 用于去重
    existing_oa = set()
    for row in conn.execute("SELECT openalex_id FROM paper_identity WHERE openalex_id IS NOT NULL"):
        existing_oa.add(row[0])

    # 建立 openalex_id → paper_id 映射
    oa_to_pid: dict[str, str] = {}
    for row in conn.execute("SELECT paper_id, openalex_id FROM paper_identity WHERE openalex_id IS NOT NULL"):
        oa_to_pid[row[1]] = row[0]

    new_papers: list[Paper] = []
    for p in papers:
        if p.openalex_id and p.openalex_id in existing_oa:
            continue
        new_papers.append(p)
        if p.openalex_id:
            oa_to_pid[p.openalex_id] = p.paper_id
            existing_oa.add(p.openalex_id)

    # 入库 paper_identity
    for p in new_papers:
        conn.execute(
            """INSERT OR IGNORE INTO paper_identity
               (paper_id, openalex_id, title, abstract, publication_year,
                cited_by_count, source, topics_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                p.paper_id, p.openalex_id, p.title, p.abstract,
                p.publication_year, p.cited_by_count, p.source, p.topics_json,
            ),
        )
        inserted += 1

    # 入库 paper_references（仅记录双方都在库中的引用）
    for p in new_papers:
        for ref_oa_id in p.referenced_work_ids:
            cited_pid = oa_to_pid.get(ref_oa_id)
            if cited_pid:
                conn.execute(
                    "INSERT OR IGNORE INTO paper_references (citing_id, cited_id) VALUES (?, ?)",
                    (p.paper_id, cited_pid),
                )

    conn.commit()
    logger.info("入库 %d 篇论文（跳过 %d 已存在）", inserted, len(papers) - inserted)
    return inserted
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/ingest/ingest.py tests/test_ingest.py
git commit -m "feat: 论文入库 paper_identity + paper_references"
```

---

### Task 5: 嵌入计算 (S1)

**Files:**
- Create: `echelon/ingest/embedder.py`
- Create: `tests/test_embedder.py`

**参考:** V14-B `step5b_vgae.py` 中 sentence-transformers all-mpnet-base-v2 的使用方式

- [ ] **Step 1: 编写嵌入测试**

```python
# tests/test_embedder.py
"""嵌入计算测试。"""
import struct
from pathlib import Path

import pytest

from echelon.db.sqlite_ops import init_db
from echelon.ingest.embedder import compute_embeddings, blob_to_vector
from echelon.ingest.ingest import ingest_papers
from echelon.schema.paper import Paper


@pytest.fixture()
def conn_with_papers(tmp_path: Path):
    conn = init_db(tmp_path / "test.sqlite3")
    papers = [
        Paper(title="Adaptive optics", abstract="wavefront sensing", publication_year=2023),
        Paper(title="Deep learning", abstract="neural network optimization", publication_year=2023),
    ]
    ingest_papers(conn, papers)
    yield conn
    conn.close()


def test_compute_embeddings(conn_with_papers):
    computed = compute_embeddings(conn_with_papers)
    assert computed == 2

    rows = conn_with_papers.execute("SELECT count(*) FROM paper_embeddings").fetchone()[0]
    assert rows == 2


def test_embedding_dimension(conn_with_papers):
    compute_embeddings(conn_with_papers)
    row = conn_with_papers.execute("SELECT embedding FROM paper_embeddings LIMIT 1").fetchone()
    vec = blob_to_vector(row[0])
    assert len(vec) == 768


def test_compute_embeddings_skip_existing(conn_with_papers):
    compute_embeddings(conn_with_papers)
    computed = compute_embeddings(conn_with_papers)
    assert computed == 0  # 已全部计算
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_embedder.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 embedder**

```python
# echelon/ingest/embedder.py
"""all-mpnet-base-v2 嵌入计算。"""
from __future__ import annotations

import logging
import struct
import sqlite3

from echelon.config import EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


def vector_to_blob(vec: list[float]) -> bytes:
    """float32 列表 → bytes。"""
    return struct.pack(f"{len(vec)}f", *vec)


def blob_to_vector(blob: bytes) -> list[float]:
    """bytes → float32 列表。"""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def compute_embeddings(conn: sqlite3.Connection) -> int:
    """计算尚未嵌入的论文的 embedding，写入 paper_embeddings。"""
    from sentence_transformers import SentenceTransformer

    # 找出未计算嵌入的论文
    rows = conn.execute(
        """SELECT p.paper_id, p.title, p.abstract
           FROM paper_identity p
           LEFT JOIN paper_embeddings e ON p.paper_id = e.paper_id
           WHERE e.paper_id IS NULL"""
    ).fetchall()

    if not rows:
        logger.info("所有论文已有嵌入，跳过")
        return 0

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [
        f"{r['title']}. {r['abstract'] or ''}" for r in rows
    ]
    paper_ids = [r["paper_id"] for r in rows]

    computed = 0
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch_texts = texts[i : i + EMBEDDING_BATCH_SIZE]
        batch_ids = paper_ids[i : i + EMBEDDING_BATCH_SIZE]
        embeddings = model.encode(batch_texts, show_progress_bar=False)

        for pid, emb in zip(batch_ids, embeddings):
            conn.execute(
                "INSERT OR IGNORE INTO paper_embeddings (paper_id, embedding) VALUES (?, ?)",
                (pid, vector_to_blob(emb.tolist())),
            )
        computed += len(batch_texts)

    conn.commit()
    logger.info("计算 %d 篇论文嵌入", computed)
    return computed
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_embedder.py -v`
Expected: 3 tests PASS（首次运行需下载模型，约 420MB）

- [ ] **Step 5: 提交**

```bash
git add echelon/ingest/embedder.py tests/test_embedder.py
git commit -m "feat: all-mpnet-base-v2 嵌入计算 + blob 序列化"
```

---

### Task 6: LLM 客户端（多 Provider）

**Files:**
- Create: `echelon/llm/__init__.py`
- Create: `echelon/llm/client.py`
- Create: `tests/test_llm_client.py`

**参考:** V14-B `llm_client.py` (370行) 的多 Provider 工厂模式 + CostTracker

- [ ] **Step 1: 编写 LLM 客户端测试**

```python
# tests/test_llm_client.py
"""LLM 客户端测试。"""
import json
from unittest.mock import patch, MagicMock

import pytest

from echelon.llm.client import LLMClient, CostTracker


def test_extract_json_parses_valid():
    client = LLMClient.__new__(LLMClient)
    client._call = MagicMock(return_value=('{"concepts": ["optics"]}', 10, 20))
    client.provider = "mock"
    client.model = "mock"
    client.cost = CostTracker()

    result = client.extract_json("prompt")
    assert result == {"concepts": ["optics"]}


def test_extract_json_handles_markdown_fence():
    client = LLMClient.__new__(LLMClient)
    client._call = MagicMock(
        return_value=('```json\n{"concepts": ["optics"]}\n```', 10, 20)
    )
    client.provider = "mock"
    client.model = "mock"
    client.cost = CostTracker()

    result = client.extract_json("prompt")
    assert result == {"concepts": ["optics"]}


def test_cost_tracker():
    tracker = CostTracker()
    tracker.add(100, 50, "anthropic")
    tracker.add(200, 100, "anthropic")
    assert tracker.total_input == 300
    assert tracker.total_output == 150
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_llm_client.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 LLM 客户端**

```python
# echelon/llm/__init__.py

# echelon/llm/client.py
"""多 Provider LLM 抽象层。"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod

from echelon.config import LLM_PROVIDER, LLM_MODEL

logger = logging.getLogger(__name__)

# token 单价（USD/1K tokens）
_COST_TABLE = {
    "anthropic": {"input": 0.003, "output": 0.015},
    "openai": {"input": 0.005, "output": 0.015},
    "ollama": {"input": 0.0, "output": 0.0},
}


class CostTracker:
    """累计 token 消耗和费用估算。"""

    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self._provider = ""

    def add(self, input_tokens: int, output_tokens: int, provider: str):
        self.total_input += input_tokens
        self.total_output += output_tokens
        self._provider = provider

    def estimate_usd(self) -> float:
        costs = _COST_TABLE.get(self._provider, {"input": 0, "output": 0})
        return (
            self.total_input / 1000 * costs["input"]
            + self.total_output / 1000 * costs["output"]
        )


class LLMClient:
    """LLM 调用客户端，支持 Anthropic/OpenAI/Ollama。"""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or LLM_PROVIDER
        self.model = model or LLM_MODEL
        self.cost = CostTracker()
        self._init_provider()

    def _init_provider(self):
        if self.provider == "anthropic":
            import anthropic
            self._anthropic = anthropic.Anthropic()
        elif self.provider == "openai":
            import openai
            self._openai = openai.OpenAI()
        elif self.provider == "ollama":
            import httpx
            self._ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self._ollama_http = httpx.Client(timeout=120.0)

    def _call(self, prompt: str, max_tokens: int = 1024) -> tuple[str, int, int]:
        """底层 API 调用，返回 (text, input_tokens, output_tokens)。"""
        if self.provider == "anthropic":
            resp = self._anthropic.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            return text, resp.usage.input_tokens, resp.usage.output_tokens

        elif self.provider == "openai":
            resp = self._openai.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content
            usage = resp.usage
            return text, usage.prompt_tokens, usage.completion_tokens

        elif self.provider == "ollama":
            resp = self._ollama_http.post(
                f"{self._ollama_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            data = resp.json()
            return data["response"], data.get("prompt_eval_count", 0), data.get("eval_count", 0)

        raise ValueError(f"未知 provider: {self.provider}")

    def extract(self, prompt: str, max_tokens: int = 1024, retries: int = 3) -> str:
        """调用 LLM 提取文本，带指数退避重试。"""
        for attempt in range(retries):
            try:
                text, inp, out = self._call(prompt, max_tokens)
                self.cost.add(inp, out, self.provider)
                return text
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning("LLM 调用失败 (attempt %d): %s，%ds 后重试", attempt + 1, e, wait)
                time.sleep(wait)
        return ""

    def extract_json(self, prompt: str, max_tokens: int = 1024) -> dict:
        """调用 LLM 并解析 JSON 输出。"""
        text = self.extract(prompt, max_tokens)
        # 处理 ```json ... ``` 包裹
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        return json.loads(text)
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_llm_client.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/llm/ tests/test_llm_client.py
git commit -m "feat: 多 Provider LLM 客户端 (Anthropic/OpenAI/Ollama)"
```

---

### Task 7: KeyBERT 概念提取 (S5d)

**Files:**
- Create: `echelon/semnet/__init__.py`
- Create: `echelon/semnet/concept_extract.py`
- Create: `tests/test_concept_extract.py`

- [ ] **Step 1: 编写 KeyBERT 提取测试**

```python
# tests/test_concept_extract.py
"""概念提取测试。"""
import pytest

from echelon.semnet.concept_extract import extract_keybert


def test_extract_keybert_returns_concepts():
    title = "Adaptive optics for astronomical telescopes"
    abstract = "We present a novel wavefront sensing approach using deep learning for real-time correction."
    concepts = extract_keybert(title, abstract)
    assert len(concepts) > 0
    assert all(isinstance(c, dict) for c in concepts)
    assert all("name" in c and "score" in c for c in concepts)


def test_extract_keybert_handles_empty_abstract():
    concepts = extract_keybert("Some title", "")
    assert isinstance(concepts, list)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_concept_extract.py::test_extract_keybert_returns_concepts -v`
Expected: FAIL

- [ ] **Step 3: 实现 KeyBERT 提取**

```python
# echelon/semnet/__init__.py

# echelon/semnet/concept_extract.py
"""LLM + KeyBERT 双通道概念提取。"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from echelon.config import (
    KEYBERT_DIVERSITY,
    KEYBERT_TOP_N,
    LLM_MAX_CONCEPTS_PER_PAPER,
)

if TYPE_CHECKING:
    from echelon.llm.client import LLMClient

logger = logging.getLogger(__name__)

# KeyBERT 单例（延迟初始化）
_keybert_model = None


def _get_keybert():
    global _keybert_model
    if _keybert_model is None:
        from keybert import KeyBERT
        _keybert_model = KeyBERT(model="all-mpnet-base-v2")
    return _keybert_model


def extract_keybert(title: str, abstract: str) -> list[dict]:
    """KeyBERT 校验通道：MMR 多样性关键短语提取。"""
    text = f"{title}. {abstract}" if abstract else title
    if len(text.strip()) < 10:
        return []

    kw = _get_keybert()
    keywords = kw.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_mmr=True,
        diversity=KEYBERT_DIVERSITY,
        top_n=KEYBERT_TOP_N,
    )
    return [{"name": kw_text, "score": float(score), "method": "keybert"} for kw_text, score in keywords]


_CONCEPT_PROMPT = """从以下学术论文的标题和摘要中提取所有领域相关的科学概念。
概念包括：技术方法、材料/器件、物理现象、算法、应用领域。
不是仅提取关键词，而是提取有科学含义的概念短语。

标题：{title}
摘要：{abstract}

以 JSON 格式输出，包含 concepts 数组，每个元素有 name（概念名，英文小写）：
{{"concepts": ["concept1", "concept2", ...]}}

要求：
- 输出 10-20 个概念
- 英文小写，2-5 个单词的短语
- 不输出通用词（如 "method", "result", "study"）
"""


def extract_llm(title: str, abstract: str, llm: LLMClient) -> list[dict]:
    """LLM 主通道：结构化 prompt 概念提取。"""
    if not abstract:
        return []

    prompt = _CONCEPT_PROMPT.format(title=title, abstract=abstract)
    try:
        result = llm.extract_json(prompt, max_tokens=512)
        concepts = result.get("concepts", [])
        return [
            {"name": c.lower().strip(), "score": 1.0, "method": "llm"}
            for c in concepts[:LLM_MAX_CONCEPTS_PER_PAPER]
            if isinstance(c, str) and len(c.strip()) >= 3
        ]
    except Exception as e:
        logger.warning("LLM 概念提取失败: %s", e)
        return []
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_concept_extract.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/semnet/ tests/test_concept_extract.py
git commit -m "feat: KeyBERT + LLM 双通道概念提取"
```

---

### Task 8: LLM 概念提取测试补充

**Files:**
- Modify: `tests/test_concept_extract.py`

- [ ] **Step 1: 补充 LLM 提取测试（mock）**

在 `tests/test_concept_extract.py` 末尾追加：

```python
from unittest.mock import MagicMock
from echelon.semnet.concept_extract import extract_llm


def test_extract_llm_returns_concepts():
    mock_llm = MagicMock()
    mock_llm.extract_json.return_value = {
        "concepts": ["adaptive optics", "wavefront sensing", "deformable mirror"]
    }
    concepts = extract_llm(
        "Adaptive optics for microscopy",
        "We use wavefront sensing with deformable mirrors.",
        mock_llm,
    )
    assert len(concepts) == 3
    assert concepts[0]["name"] == "adaptive optics"
    assert concepts[0]["method"] == "llm"


def test_extract_llm_handles_failure():
    mock_llm = MagicMock()
    mock_llm.extract_json.side_effect = Exception("API error")
    concepts = extract_llm("Title", "Abstract", mock_llm)
    assert concepts == []
```

- [ ] **Step 2: 运行测试验证**

Run: `python -m pytest tests/test_concept_extract.py -v`
Expected: 4 tests PASS

- [ ] **Step 3: 提交**

```bash
git add tests/test_concept_extract.py
git commit -m "test: LLM 概念提取 mock 测试"
```

---

### Task 9: 双通道合并 + 同义词归一化 (S5d)

**Files:**
- Create: `echelon/semnet/concept_normalize.py`
- Create: `tests/test_concept_normalize.py`

- [ ] **Step 1: 编写归一化测试**

```python
# tests/test_concept_normalize.py
"""概念合并 + 同义词归一化测试。"""
import pytest

from echelon.semnet.concept_normalize import merge_concepts, normalize_synonyms


def test_merge_concepts_union():
    llm = [
        {"name": "adaptive optics", "score": 1.0, "method": "llm"},
        {"name": "wavefront sensing", "score": 1.0, "method": "llm"},
    ]
    keybert = [
        {"name": "adaptive optics system", "score": 0.7, "method": "keybert"},
        {"name": "telescope", "score": 0.5, "method": "keybert"},
    ]
    merged = merge_concepts(llm, keybert, dedup_threshold=0.90)
    names = {c["name"] for c in merged}
    assert "telescope" in names
    assert len(merged) >= 3  # 至少 3 个唯一概念


def test_normalize_synonyms_clusters():
    concepts = [
        "adaptive optics",
        "adaptive optical system",
        "deep learning",
        "deep neural network",
        "wavefront sensing",
    ]
    normalized = normalize_synonyms(concepts, distance_threshold=0.15)
    # 应该有聚类：adaptive optics 系列，deep learning 系列
    assert len(normalized) <= len(concepts)
    # 每个 cluster 有 canonical_name
    for entry in normalized:
        assert "canonical_name" in entry
        assert "members" in entry


def test_normalize_synonyms_preserves_distinct():
    concepts = ["photonic crystal", "quantum computing", "fluid dynamics"]
    normalized = normalize_synonyms(concepts, distance_threshold=0.15)
    assert len(normalized) == 3  # 完全不同的概念不应合并
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_concept_normalize.py -v`
Expected: FAIL

- [ ] **Step 3: 实现合并 + 归一化**

```python
# echelon/semnet/concept_normalize.py
"""双通道合并 + 嵌入聚类同义词归一化。"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from echelon.config import CONCEPT_DEDUP_SIMILARITY, NORMALIZE_COSINE_THRESHOLD, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_sbert = None


def _get_sbert() -> SentenceTransformer:
    global _sbert
    if _sbert is None:
        _sbert = SentenceTransformer(EMBEDDING_MODEL)
    return _sbert


def _lemmatize(text: str) -> str:
    """小写化 + spaCy lemmatization。"""
    text = text.lower().strip()
    try:
        import spacy
        nlp = spacy.blank("en")
        doc = nlp(text)
        return " ".join(token.lemma_ for token in doc)
    except Exception:
        return text


def merge_concepts(
    llm_concepts: list[dict],
    keybert_concepts: list[dict],
    dedup_threshold: float = CONCEPT_DEDUP_SIMILARITY,
) -> list[dict]:
    """双通道合并：取并集，嵌入余弦相似度去重。"""
    all_concepts = llm_concepts + keybert_concepts
    if not all_concepts:
        return []

    names = [c["name"] for c in all_concepts]
    model = _get_sbert()
    embeddings = model.encode(names, show_progress_bar=False)

    # 贪心去重：按顺序保留，若与已保留概念相似度 >= threshold 则跳过
    kept: list[int] = []
    for i in range(len(all_concepts)):
        is_dup = False
        for j in kept:
            sim = float(np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
            ))
            if sim >= dedup_threshold:
                # 如果来自不同通道，标记为 "both"
                if all_concepts[j]["method"] != all_concepts[i]["method"]:
                    all_concepts[j]["method"] = "both"
                    all_concepts[j]["score"] = max(all_concepts[j]["score"], all_concepts[i]["score"])
                is_dup = True
                break
        if not is_dup:
            kept.append(i)

    return [all_concepts[i] for i in kept]


def normalize_synonyms(
    concept_names: list[str],
    distance_threshold: float = NORMALIZE_COSINE_THRESHOLD,
) -> list[dict]:
    """嵌入聚类同义词归一化。"""
    if not concept_names:
        return []

    # lemmatization 前处理
    lemmatized = [_lemmatize(c) for c in concept_names]

    # 嵌入
    model = _get_sbert()
    embeddings = model.encode(lemmatized, show_progress_bar=False)

    if len(concept_names) == 1:
        return [{"canonical_name": concept_names[0], "members": [concept_names[0]], "embedding": embeddings[0]}]

    # 层次聚类
    dist_matrix = cosine_distances(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist_matrix)

    # 按聚类分组
    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    result = []
    for label, indices in clusters.items():
        members = [concept_names[i] for i in indices]
        # 选择出现频率最高的作为 canonical_name
        counter = Counter(members)
        canonical = counter.most_common(1)[0][0]
        # 聚类中心嵌入
        cluster_emb = np.mean([embeddings[i] for i in indices], axis=0)
        result.append({
            "canonical_name": canonical,
            "members": members,
            "embedding": cluster_emb,
        })

    logger.info("归一化 %d 个概念 → %d 个聚类", len(concept_names), len(result))
    return result
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_concept_normalize.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/semnet/concept_normalize.py tests/test_concept_normalize.py
git commit -m "feat: 双通道合并 + 嵌入聚类同义词归一化"
```

---

### Task 10: 概念共现网络 (S5e)

**Files:**
- Create: `echelon/semnet/concept_network.py`
- Create: `tests/test_concept_network.py`

- [ ] **Step 1: 编写共现网络测试**

```python
# tests/test_concept_network.py
"""概念共现网络构建测试。"""
from pathlib import Path

import pytest

from echelon.db.sqlite_ops import init_db
from echelon.semnet.concept_network import build_cooccurrence_network


@pytest.fixture()
def conn_with_concepts(tmp_path: Path):
    conn = init_db(tmp_path / "test.sqlite3")
    # 插入论文
    conn.executemany(
        "INSERT INTO paper_identity (paper_id, title, publication_year) VALUES (?, ?, ?)",
        [("P1", "Paper1", 2022), ("P2", "Paper2", 2023), ("P3", "Paper3", 2023)],
    )
    # 插入概念
    conn.executemany(
        "INSERT INTO concept_nodes (concept_id, canonical_name, first_seen_year) VALUES (?, ?, ?)",
        [("C1", "optics", 2022), ("C2", "deep learning", 2022), ("C3", "wavefront", 2023)],
    )
    # 插入论文-概念关联
    conn.executemany(
        "INSERT INTO paper_concepts (paper_id, concept_id, score, method) VALUES (?, ?, ?, ?)",
        [
            ("P1", "C1", 1.0, "llm"),  # P1: optics + deep learning
            ("P1", "C2", 0.8, "keybert"),
            ("P2", "C1", 1.0, "llm"),  # P2: optics + wavefront
            ("P2", "C3", 0.9, "llm"),
            ("P3", "C2", 1.0, "llm"),  # P3: deep learning + wavefront
            ("P3", "C3", 0.7, "keybert"),
        ],
    )
    conn.commit()
    yield conn
    conn.close()


def test_build_cooccurrence_network(conn_with_concepts):
    edges = build_cooccurrence_network(conn_with_concepts)
    assert len(edges) > 0

    # 检查 DB 已写入
    rows = conn_with_concepts.execute("SELECT count(*) FROM concept_cooccurrence").fetchone()[0]
    assert rows == len(edges)


def test_cooccurrence_year_slicing(conn_with_concepts):
    build_cooccurrence_network(conn_with_concepts)

    # 2022 年：C1-C2 共现（P1）
    r2022 = conn_with_concepts.execute(
        "SELECT * FROM concept_cooccurrence WHERE year = 2022"
    ).fetchall()
    assert len(r2022) >= 1

    # 2023 年：C1-C3 共现（P2），C2-C3 共现（P3）
    r2023 = conn_with_concepts.execute(
        "SELECT * FROM concept_cooccurrence WHERE year = 2023"
    ).fetchall()
    assert len(r2023) >= 2


def test_cooccurrence_symmetric(conn_with_concepts):
    build_cooccurrence_network(conn_with_concepts)
    # 约定 concept_a < concept_b（字典序）确保唯一
    rows = conn_with_concepts.execute(
        "SELECT concept_a, concept_b FROM concept_cooccurrence"
    ).fetchall()
    for row in rows:
        assert row[0] < row[1], f"非规范序：{row[0]} >= {row[1]}"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_concept_network.py -v`
Expected: FAIL

- [ ] **Step 3: 实现共现网络**

```python
# echelon/semnet/concept_network.py
"""概念共现网络构建（年份切片 + 时间衰减）。"""
from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from itertools import combinations

from echelon.config import COOCCURRENCE_TIME_DECAY

logger = logging.getLogger(__name__)


def build_cooccurrence_network(conn: sqlite3.Connection) -> list[dict]:
    """从 paper_concepts 构建概念共现网络，写入 concept_cooccurrence 表。"""
    # 查询每篇论文的概念列表 + 年份
    rows = conn.execute(
        """SELECT pc.paper_id, pc.concept_id, pi.publication_year
           FROM paper_concepts pc
           JOIN paper_identity pi ON pc.paper_id = pi.paper_id
           ORDER BY pc.paper_id"""
    ).fetchall()

    # 按论文分组
    paper_concepts: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in rows:
        paper_concepts[row["paper_id"]].append((row["concept_id"], row["publication_year"]))

    # 统计共现（按年份切片）
    cooccurrence: dict[tuple[str, str, int], int] = defaultdict(int)
    for paper_id, concepts_years in paper_concepts.items():
        concept_ids = [c[0] for c in concepts_years]
        year = concepts_years[0][1] if concepts_years else 0

        for c_a, c_b in combinations(sorted(set(concept_ids)), 2):
            # 规范序：字典序小的在前
            key = (min(c_a, c_b), max(c_a, c_b), year)
            cooccurrence[key] += 1

    # 计算当前年份（用于时间衰减）
    max_year = max((y for _, _, y in cooccurrence), default=2025)

    # 写入 DB
    conn.execute("DELETE FROM concept_cooccurrence")
    edges = []
    for (c_a, c_b, year), count in cooccurrence.items():
        decay = COOCCURRENCE_TIME_DECAY ** (max_year - year)
        weight = count * decay
        conn.execute(
            "INSERT INTO concept_cooccurrence (concept_a, concept_b, year, weight) VALUES (?, ?, ?, ?)",
            (c_a, c_b, year, weight),
        )
        edges.append({"concept_a": c_a, "concept_b": c_b, "year": year, "weight": weight})

    conn.commit()
    logger.info("构建共现网络：%d 条边（%d 个年份切片）", len(edges), len({e["year"] for e in edges}))
    return edges
```

- [ ] **Step 4: 运行测试验证**

Run: `python -m pytest tests/test_concept_network.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: 提交**

```bash
git add echelon/semnet/concept_network.py tests/test_concept_network.py
git commit -m "feat: 概念共现网络构建（年份切片 + 时间衰减）"
```

---

### Task 11: Phase 1 端到端集成 + 流水线编排

**Files:**
- Create: `echelon/run.py`
- Create: `tests/test_pipeline_phase1.py`

- [ ] **Step 1: 编写端到端集成测试**

```python
# tests/test_pipeline_phase1.py
"""Phase 1 端到端集成测试（使用 mock OpenAlex + mock LLM）。"""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from echelon.db.sqlite_ops import init_db
from echelon.run import run_phase1


def _mock_openalex_works() -> list[dict]:
    return [
        {
            "id": f"https://openalex.org/W{i:03d}",
            "title": f"Paper {i}: {'Adaptive Optics' if i % 2 == 0 else 'Deep Learning'} Research",
            "abstract_inverted_index": {
                "This": [0], "paper": [1], "studies": [2],
                ("adaptive" if i % 2 == 0 else "deep"): [3],
                ("optics" if i % 2 == 0 else "learning"): [4],
            },
            "publication_date": f"202{2 + i % 3}-06-15",
            "cited_by_count": i * 5,
            "primary_topic": {
                "id": "https://openalex.org/T10245",
                "display_name": "Optical Coherence Tomography",
            },
            "referenced_works": [f"https://openalex.org/W{max(0,i-1):03d}"] if i > 0 else [],
        }
        for i in range(10)
    ]


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.sqlite3"


@pytest.mark.asyncio
async def test_phase1_pipeline(db_path: Path):
    mock_works = _mock_openalex_works()

    with (
        patch("echelon.run.fetch_works_by_topics") as mock_fetch,
        patch("echelon.run.LLMClient") as MockLLM,
    ):
        # mock OpenAlex API
        async def fake_fetch(**kwargs):
            for w in mock_works:
                yield w

        mock_fetch.return_value = fake_fetch()

        # mock LLM
        mock_llm_inst = MagicMock()
        mock_llm_inst.extract_json.return_value = {
            "concepts": ["adaptive optics", "wavefront sensing", "deep learning"]
        }
        mock_llm_inst.cost = MagicMock()
        mock_llm_inst.cost.estimate_usd.return_value = 0.01
        mock_llm_inst.cost.total_input = 100
        mock_llm_inst.cost.total_output = 50
        MockLLM.return_value = mock_llm_inst

        stats = await run_phase1(db_path)

    conn = init_db(db_path)

    # 验证：论文入库
    paper_count = conn.execute("SELECT count(*) FROM paper_identity").fetchone()[0]
    assert paper_count == 10, f"预期 10 篇，实际 {paper_count}"

    # 验证：嵌入计算
    emb_count = conn.execute("SELECT count(*) FROM paper_embeddings").fetchone()[0]
    assert emb_count == 10

    # 验证：概念提取
    concept_count = conn.execute("SELECT count(*) FROM concept_nodes").fetchone()[0]
    assert concept_count > 0, "应有概念节点"

    # 验证：论文-概念关联
    pc_count = conn.execute("SELECT count(*) FROM paper_concepts").fetchone()[0]
    assert pc_count > 0

    # 验证：共现网络
    cooc_count = conn.execute("SELECT count(*) FROM concept_cooccurrence").fetchone()[0]
    assert cooc_count > 0

    conn.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pipeline_phase1.py -v`
Expected: FAIL — `No module named 'echelon.run'`

- [ ] **Step 3: 实现 run.py 流水线编排**

```python
# echelon/run.py
"""主入口：端到端流水线编排。"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from echelon.config import (
    DB_PATH,
    OPENALEX_PILOT_TOPICS,
    OPENALEX_PILOT_SINCE,
    OPENALEX_PILOT_UNTIL,
    LIMIT,
)
from echelon.db.sqlite_ops import init_db, get_conn
from echelon.ingest.fetcher import fetch_works_by_topics
from echelon.ingest.ingest import ingest_papers
from echelon.ingest.embedder import compute_embeddings
from echelon.llm.client import LLMClient
from echelon.schema.paper import Paper
from echelon.semnet.concept_extract import extract_keybert, extract_llm
from echelon.semnet.concept_normalize import merge_concepts, normalize_synonyms
from echelon.semnet.concept_network import build_cooccurrence_network

logger = logging.getLogger(__name__)


async def run_phase1(db_path: Path = DB_PATH) -> dict:
    """Phase 1 流水线：S0 爬取 → S1 入库 → S5d 概念提取 → 共现网络。"""
    conn = init_db(db_path)
    stats = {}

    # ── S0 数据爬取 ──
    logger.info("=== S0 数据爬取 ===")
    raw_works: list[dict] = []
    async for work in fetch_works_by_topics(
        topic_ids=OPENALEX_PILOT_TOPICS,
        since=OPENALEX_PILOT_SINCE,
        until=OPENALEX_PILOT_UNTIL,
        max_results=LIMIT,
    ):
        raw_works.append(work)
    stats["s0_fetched"] = len(raw_works)
    logger.info("爬取 %d 篇论文", len(raw_works))

    # ── S1 数据入库 ──
    logger.info("=== S1 数据入库 ===")
    papers = [Paper.from_openalex(w) for w in raw_works]
    stats["s1_ingested"] = ingest_papers(conn, papers)

    # ── S1 嵌入计算 ──
    logger.info("=== S1 嵌入计算 ===")
    stats["s1_embeddings"] = compute_embeddings(conn)

    # ── S5d 概念提取 ──
    logger.info("=== S5d 概念提取 ===")
    llm = LLMClient()

    all_concepts: list[str] = []
    paper_concept_pairs: list[tuple[str, str, float, str]] = []

    rows = conn.execute("SELECT paper_id, title, abstract FROM paper_identity").fetchall()
    for row in rows:
        pid, title, abstract = row["paper_id"], row["title"], row["abstract"] or ""

        # 双通道提取
        kb_concepts = extract_keybert(title, abstract)
        llm_concepts = extract_llm(title, abstract, llm)
        merged = merge_concepts(llm_concepts, kb_concepts)

        for c in merged:
            all_concepts.append(c["name"])
            paper_concept_pairs.append((pid, c["name"], c["score"], c["method"]))

    # ── 同义词归一化 ──
    logger.info("=== S5d 同义词归一化 ===")
    unique_concepts = list(set(all_concepts))
    clusters = normalize_synonyms(unique_concepts)

    # 写入 concept_nodes
    import ulid as _ulid
    name_to_cid: dict[str, str] = {}
    for cluster in clusters:
        cid = str(_ulid.new())
        canonical = cluster["canonical_name"]
        emb_blob = None
        if "embedding" in cluster:
            from echelon.ingest.embedder import vector_to_blob
            emb_blob = vector_to_blob(cluster["embedding"].tolist())

        # 查找 first_seen_year
        member_names = cluster["members"]
        for m in member_names:
            name_to_cid[m] = cid

        conn.execute(
            """INSERT OR IGNORE INTO concept_nodes
               (concept_id, canonical_name, total_papers, embedding)
               VALUES (?, ?, ?, ?)""",
            (cid, canonical, 0, emb_blob),
        )

    # 写入 paper_concepts（用归一化后的 concept_id）
    for pid, cname, score, method in paper_concept_pairs:
        cid = name_to_cid.get(cname)
        if cid:
            conn.execute(
                "INSERT OR IGNORE INTO paper_concepts (paper_id, concept_id, score, method) VALUES (?, ?, ?, ?)",
                (pid, cid, score, method),
            )

    # 更新 concept_nodes 的 total_papers 和 first_seen_year
    conn.execute(
        """UPDATE concept_nodes SET total_papers = (
             SELECT count(DISTINCT paper_id) FROM paper_concepts WHERE concept_id = concept_nodes.concept_id
           )"""
    )
    conn.execute(
        """UPDATE concept_nodes SET first_seen_year = (
             SELECT min(pi.publication_year) FROM paper_concepts pc
             JOIN paper_identity pi ON pc.paper_id = pi.paper_id
             WHERE pc.concept_id = concept_nodes.concept_id
           )"""
    )
    conn.commit()

    stats["s5d_concepts"] = conn.execute("SELECT count(*) FROM concept_nodes").fetchone()[0]
    stats["s5d_paper_concepts"] = conn.execute("SELECT count(*) FROM paper_concepts").fetchone()[0]

    # ── S5e 共现网络 ──
    logger.info("=== S5e 共现网络 ===")
    edges = build_cooccurrence_network(conn)
    stats["s5e_cooccurrence_edges"] = len(edges)

    conn.close()

    # ── 输出统计 ──
    logger.info("=== Phase 1 完成 ===")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(run_phase1())
```

- [ ] **Step 4: 运行端到端测试**

Run: `python -m pytest tests/test_pipeline_phase1.py -v`
Expected: 1 test PASS

- [ ] **Step 5: 运行全部测试确认无回归**

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS（约 16 tests）

- [ ] **Step 6: 提交**

```bash
git add echelon/run.py tests/test_pipeline_phase1.py
git commit -m "feat: Phase 1 端到端流水线编排 (S0→S1→S5d→S5e)"
```

---

## Phase 1 验证标准

完成 Task 11 后，执行实际数据验证：

```bash
# 设置环境变量
export OPENALEX_MAILTO="your@email.com"
export LLM_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-..."

# 运行 Phase 1（200 篇 Pilot）
cd /Users/lihaisen/PycharmProjects/Echelon
python -m echelon.run
```

**必须达到的指标：**

| 指标 | 目标 | 检查 SQL |
|------|------|---------|
| paper_identity 入库 | ≥ 200 篇 | `SELECT count(*) FROM paper_identity` |
| paper_embeddings | = paper_identity 数量 | `SELECT count(*) FROM paper_embeddings` |
| concept_nodes | ≥ 100 个概念 | `SELECT count(*) FROM concept_nodes` |
| paper_concepts 覆盖率 | ≥ 90% 论文有概念 | `SELECT count(DISTINCT paper_id) FROM paper_concepts` / 总论文数 |
| concept_cooccurrence | > 0 边 | `SELECT count(*) FROM concept_cooccurrence` |
