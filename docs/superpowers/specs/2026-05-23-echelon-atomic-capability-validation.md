# Echelon 原子能力可行性验证报告 v1.0

> **版本**: 1.0  
> **日期**: 2026-05-23  
> **目的**: 将 Spec 中所有技术拆解为 28 个原子能力，逐一搜索顶级论文验证可行性与可实现性  
> **标准**: 仅接受正式发表论文（非预印本），标注期刊/会议等级

---

## 验证总览

| 状态 | 数量 | 说明 |
|------|------|------|
| ✅ 强论文支撑 | 20 | 有 2+ 篇 A*/A 级论文直接验证 |
| ⚠️ 间接/部分支撑 | 5 | 有论文支撑但非直接场景，或仅 1 篇直接论文 |
| 🔧 工具级（无需论文） | 3 | 成熟工程实践，不构成学术创新点 |
| ❌ 无支撑 | 0 | — |

---

## 一、数据采集层（S0/S1）

### A1. OpenAlex API 大规模学术数据爬取 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Culbert et al., "Reference Coverage Analysis of OpenAlex Compared to Web of Science and Scopus" | Scientometrics 2025 | GESIS Leibniz Institute; Göttingen; DZHW | JCR Q1 | 1678 万篇共有出版物上，OpenAlex 参考文献覆盖率与 WoS/Scopus 相当 |
| Thelwall & Jiang, "Is OpenAlex Suitable for Research Quality Evaluation?" | JASIST 2025 | University of Sheffield | CCF-B / JCR Q1 | 2860 万篇论文分析，OpenAlex 引用计数优于 Scopus |
| Forchino et al., "The OpenAlex Database in Review" | Journal of Informetrics 2026 | University of Granada | CORE A / JCR Q1 | 系统综述 146 篇使用 OpenAlex 的文献，确认其作为商业数据库开放替代品的可行性 |

**结论**：完全可行。覆盖率与商业数据库可比，免费开放 API，已被学术界广泛采用。

---

### A2. Pydantic v2 数据验证 + ULID 主键 🔧

**结论**：无需论文验证。属于工程实践层面的技术选型：
- **Pydantic v2**：Python 生态最快验证框架（Rust 核心），被 FastAPI 等主流框架采用（GitHub 24k+ Stars）
- **ULID**：标识符规范，时序可排序 + 低碰撞率，工业实践充分验证

---

### A3. all-mpnet-base-v2 (Sentence-BERT) 学术文本嵌入 ⚠️

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" | EMNLP 2019 | TU Darmstadt | CCF-B | 奠基论文，引用 15000+，STS benchmark SOTA |
| Muennighoff et al., "MTEB: Massive Text Embedding Benchmark" | EACL 2023 | Hugging Face; Cohere | CORE A | 58 数据集基准，all-mpnet-base-v2 为顶级通用嵌入模型之一 |
| Singh et al., "SciRepEval / SPECTER2" | EMNLP 2023 | AI2; Northwestern; Yale | CCF-B | 学术文本专用模型 SPECTER2 优于通用 SBERT，但聚类任务中轻量级模型（MPNet）反而更优 |

**结论**：可行。通用场景表现良好。若后期需更高学术文本精度，可升级至 SPECTER2。

**⚠️ 风险提示**：all-mpnet-base-v2 为间接验证（无独立论文），但其基础架构和基准评估充分。

---

## 二、引文图谱层（S2-S4）

### A4. SPC (Search Path Count) 主路径分析 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Hummon & Doreian, "Connectivity in a Citation Network" | Social Networks 1989 | University of Pittsburgh | SSCI 核心 | 奠基论文，定义 NPPC/SPLC/SPNP 三种弧权重，DNA 研究引文网络验证 |
| Liu & Lu, "An Integrated Approach for Main Path Analysis" | JASIST 2012 | NTUST | A* | 提出 key-route 搜索，揭示 h-index 发展的"分散-收敛-分散"结构 |
| Jiang et al., "Main Path Analysis on Cyclic Citation Networks" | JASIST 2020 | Coventry University | A* | SimSPC 算法处理含环引文网络，引入"主路径树"数据结构 |

**结论**：完全可行。35 年成熟方法，持续演进，算法公开可实现。

---

### A5. Bibliographic Coupling + Co-citation 引文图构建 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Kessler, "Bibliographic Coupling Between Scientific Papers" | American Documentation 1963 | MIT | 奠基 | 首次定义文献耦合概念 |
| Small, "Co-citation in the Scientific Literature" | JASIS 1973 | ISI | 奠基 | 首次提出共被引分析方法 |
| Boyack & Klavans, "Which Citation Approach Represents the Research Front Most Accurately?" | JASIST 2010 | Sandia National Labs; SciTech Strategies | A* | 215 万篇文献验证：BC 识别当前前沿最优，Co-citation 揭示历史结构，两者互补 |

**结论**：完全可行。60 年经典方法，两者互补使用已被系统验证。

---

### A6. KeystoneScore 关键论文评分 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Mariani et al., "Identification of Milestone Papers Through Time-balanced Network Centrality" | J. Informetrics 2016 | University of Fribourg | A / JCR Q1 | 449,935 篇 APS 论文：PageRank + 时间平衡显著优于单纯引用计数 |
| Wang et al., "The Local Structure of Citation Networks Uncovers Expert-selected Milestone Papers" | J. Informetrics 2021 | UESTC; U. Fribourg | A / JCR Q1 | 全局排名指标（PageRank）优于局部指标（引用计数） |
| Diallo et al., "Identifying Key Papers Within a Journal via Network Centrality Measures" | Scientometrics 2016 | Old Dominion University | JCR Q1 | 特征向量中心性作为论文重要性过滤器最有效 |

**结论**：完全可行。多种 centrality 指标组合评估论文重要性优于单一引用计数。

---

### A7. SciBERT 引用功能分类 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Beltagy et al., "SciBERT: A Pretrained Language Model for Scientific Text" | EMNLP 2019 | AI2 | CCF-B / A* | 118 万篇科学论文预训练，多项科学文本 NLP 任务 SOTA |
| Cohan et al., "Structural Scaffolds for Citation Intent Classification" | NAACL 2019 | AI2 | A* | ACL-ARC 数据集 F1 提升 13.3%，发布 SciCite 数据集 |
| Jurgens et al., "Measuring the Evolution of a Scientific Field through Citation Frames" | TACL 2018 | U. Michigan; Stanford | A* | 定义 6 类引用功能，证明引用分类可揭示学科演化趋势 |

**结论**：完全可行。数据集 + 预训练模型 + 工具链全部开源就绪。

---

## 三、论文级预测层（S5a-S5c）

### A8. VGAE 论文级链接预测 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Kipf & Welling, "Variational Graph Auto-Encoders" | NeurIPS 2016 Workshop | U. Amsterdam | A* Workshop | 奠基论文，引用 6000+，Cora/Citeseer/Pubmed AUC ~91% |
| Guo et al., "Multi-Scale VGAE for Link Prediction" (MSVGAE) | WSDM 2022 | 山西大学 | A | 多尺度分布学习超越标准 VGAE |
| Ma et al., "Reconsidering the Performance of GAE in Link Prediction" | CIKM 2025 **最佳论文** | 北京大学 | A | 精调 GAE 在 ogbl-ppa 达 SOTA (Hits@100=78.41%)，超越所有复杂模型 |

**结论**：完全可行。VGAE 是引文网络链接预测的成熟基线，且 CIKM 2025 证明经优化后仍具 SOTA 竞争力。

---

### A9. D-VGAE 解耦变分图自编码器 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Cho, "Decoupled Variational Graph Autoencoder for Link Prediction" | WWW 2024 | Chung-Ang University | A* | 解耦 homophily/popularity，解决内积解码器根本缺陷 |

**结论**：可行。A* 论文直接验证。但注意 CIKM 2025 表明调优后的标准 GAE 也能达到 SOTA，D-VGAE 的增量价值需在项目数据上验证。

---

### A10. Limitation Tracking 文本驱动限制追踪 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Xu et al., "Can LLMs Identify Critical Limitations within Scientific Research?" (LimitGen) | ACL 2025 | Yale University; TCS Research | A* | 首个系统性基准，RAG 显著提升 LLM 识别限制的能力 |
| Al Azher et al., "BAGELS: Benchmarking Automated Generation and Extraction of Limitations" | EMNLP 2025 Findings | Northern Illinois U.; U. North Texas | A* | 从 ACL/NeurIPS/PeerJ 论文提取限制，基于 RAG 的生成方法 |
| Lan et al., "Automatic Categorization of Self-Acknowledged Limitations in RCT Publications" | J. Biomedical Informatics 2024 | UIUC; Amsterdam UMC | SCI Q1 | PubMedBERT 微调，限制句子检测 F1=0.70 |

**结论**：完全可行。2024-2025 年集中涌现，LLM + NLP 方法均已验证，标准化基准已就绪。

---

## 四、概念提取层（S5d）

### A11. LLM 结构化 prompt 概念提取 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Dagdelen et al., "Structured Information Extraction from Scientific Text with LLMs" | Nature Communications 2024 | Lawrence Berkeley Lab; UC Berkeley | SCI Q1 顶刊 | LLM 微调可从复杂科学文本提取结构化 JSON，~500 标注即高精度 |
| Labrak et al., "Zero-shot and Few-shot Study of Instruction-Finetuned LLMs Applied to Clinical Tasks" | LREC-COLING 2024 | Avignon U.; Nantes U./CNRS | CCF-B / CORE A | LLM 在生物医学 NLP 零/少样本任务接近 SOTA |

**结论**：完全可行。Nature Communications 直接验证技术路径。

---

### A12. KeyBERT (MMR diversity=0.7) 关键短语提取 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Bennani-Smires et al., "EmbedRank: Simple Unsupervised Keyphrase Extraction using Sentence Embeddings" | CoNLL 2018 | Swisscom AG; EPFL | CORE A | KeyBERT MMR 机制的理论基础，用户研究证明高多样性更受偏好 |
| Giarelis & Karacapilidis, "Deep Learning and Embeddings-based Approaches for Keyphrase Extraction" | KAIS 2024 | U. Patras | CCF-B / SCI Q2 | 系统综述，KeyBERT 在 KDD/WWW 数据集取最佳分数 |

**结论**：完全可行。MMR 多样性机制有理论基础，KeyBERT 有基准评估。diversity=0.7 需在目标领域数据调优。

---

### A13. 双通道合并（嵌入余弦相似度 ≥ 0.90 去重） ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Zeakis et al., "Pre-trained Embeddings for Entity Resolution" | PVLDB 2023 | U. Athens | CCF-A / CORE A* | 12 种语言模型 × 17 个基准，系统验证嵌入用于实体匹配 |
| Abbas et al., "SemDeDup: Data-efficient Learning at Web-scale through Semantic Deduplication" | ICLR 2023 Workshop | Meta AI (FAIR); Stanford | A* Workshop | 嵌入余弦阈值去重，50% 数据可移除且性能无损 |

**结论**：完全可行。嵌入语义去重在 VLDB/ICLR 验证。0.90 阈值合理，需在领域数据标定。

---

### A14. 嵌入聚类同义词归一化（层次聚类 + cosine threshold=0.15） ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Liu et al., "SapBERT: Self-Alignment Pretraining for Biomedical Entity Representations" | NAACL 2021 | U. Cambridge | CORE A | 利用 UMLS 同义词集在嵌入空间聚类对齐，6 个基准 SOTA |
| Angell et al., "Clustering-based Inference for Biomedical Entity Linking" | NAACL 2021 | UMass Amherst | CORE A | 实体链接建模为监督聚类，准确率提升 3.0% |

**结论**：完全可行。嵌入聚类用于同义词归一化有 NLP 顶会充分支撑。threshold=0.15 需在光学/AI 术语上验证。

---

### A15. spaCy lemmatization 文本前处理 🔧

**结论**：无需论文验证。成熟工具级技术：
- spaCy v3 EditTreeLemmatizer 在 Universal Dependencies 基准达 95%+ 准确率
- GitHub 60k+ Stars，被数千篇论文引用
- 英文科学术语的 lemmatization 完全可靠

---

## 五、概念网络层（S5e）

### A16. 概念共现网络构建（年份切片 + 时间衰减 λ=0.8） ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Huang et al., "Tracking the Dynamics of Co-word Networks for Emerging Topic Identification" | TFSC 2021 | 北京理工大学 | SSCI Q1 / ABS 3★ | 动态共词网络 + 年份切片 + 链接预测识别新兴科技主题 |
| Li et al., "Predicting Co-word Links via Heterogeneous Graph Convolutional Networks" | Scientific Reports 2025 | 长春理工大学 | SCI Q1 (Nature) | 共词网络建模为异质图，GCN 共词链接预测超越传统方法 |

**结论**：完全可行。时序共词网络是科学计量学成熟方法。λ=0.8 为合理默认值，可通过网格搜索优化。

---

### A17. 拓扑特征工程（30+ 特征用于链接预测） ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Ran et al., "The Maximum Capability of a Topological Feature in Link Prediction" | PNAS Nexus 2024 | 西南大学; 北京师范大学 | SCI Q1 (PNAS) | 首次理论证明拓扑特征在链接预测中的预测力上界 |
| Wu et al., "Link Prediction on Complex Networks: An Experimental Survey" | DSE 2022 | 南开大学; UMass Lowell | SCI Q2 | 36 数据集系统比较局部/全局拓扑特征和嵌入方法 |

**结论**：完全可行。拓扑特征在链接预测中有理论证明和实验验证。

---

### A18. Leiden 社区检测 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Traag et al., "From Louvain to Leiden: Guaranteeing Well-Connected Communities" | Scientific Reports 2019 | CWTS, Leiden University | SCI Q1 (Nature) | Louvain 25% 社区非良连通；Leiden 保证良连通且更快更优 |
| Sahu et al., "GVE-Leiden: Fast Leiden Algorithm in Shared Memory Setting" | ICPP 2024 | IIT Jodhpur | CCF-B | 并行 Leiden 达 4.03 亿边/秒，可扩展到 38 亿边 |

**结论**：完全可行。Leiden 是 Louvain 的严格改进，已有高性能并行实现。

---

### A19. TransE / DistMult KGE 嵌入 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Rossi et al., "Knowledge Graph Embedding for Link Prediction: A Comparative Analysis" | ACM TKDD 2021 | Roma Tre; U. Alberta | CCF-B / SCI Q2 | 13 模型 × 6 数据集系统评估，TransE/DistMult 各有优势场景 |
| Safavi & Koutra, "CoDEx: A Comprehensive KG Completion Benchmark" | EMNLP 2020 | U. Michigan | CCF-B | CoDEx 基准比 FB15k-237 更具挑战性 |
| Nayyeri et al., "Trans4E: Link Prediction on Scholarly Knowledge Graphs" | Neurocomputing 2021 | U. Bonn; Open University | SCI Q1 | TransE 扩展专门针对学术知识图谱，处理 N:M 关系 |

**结论**：完全可行。TransE/DistMult 为 KGE 基础模型，Trans4E 直接验证了在学术概念图谱上的适用性。

---

### A20. SHAP 可解释性分析 ⚠️

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" | NeurIPS 2017 | U. Washington | CCF-A / A* | SHAP 奠基论文，引用 40000+，适用于任何 ML 模型 |
| Akkas & Azad, "GNNShap: Scalable and Accurate GNN Explanation using Shapley Values" | WWW 2024 | Indiana U. Bloomington | CCF-A / A* | Shapley 值可有效用于图神经网络可解释性 |

**结论**：可行。SHAP 方法论极成熟，GNNShap 验证了在图任务中的适用性。但"SHAP + 链接预测特征归因"的专门论文目前仅有预印本，技术上无障碍。

---

## 六、ML 预测层

### A21. XGBoost + LightGBM GBDT 集成链接预测 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Grinsztajn et al., "Why do tree-based models still outperform deep learning on typical tabular data?" | NeurIPS 2022 | INRIA Saclay; Sorbonne | A* | 45 数据集：树模型在中等规模表格数据优于 DL，对无信息特征更鲁棒 |
| McElfresh et al., "When Do Neural Nets Outperform Boosted Trees on Tabular Data?" | NeurIPS 2023 | Abacus.AI; NYU; UMD | A* | 19 算法 × 176 数据集：GBDT vs NN 差异常可忽略，超参调优更重要 |

**结论**：完全可行。链接预测的手工特征本质是表格数据，GBDT 在该场景有 NeurIPS 论文验证。

---

### A22. 前馈神经网络 (100-100-10) 链接预测 ⚠️

**结论**：间接支撑。浅层 MLP 用于特征驱动链接预测的直接顶会论文少（多数聚焦 GNN），但：
- NeurIPS 2022/2023 证明 MLP 与 GBDT 在表格场景差异可忽略
- Ghasemian (PNAS 2020) 的 203 算法集成中包含浅层 NN 作为基模型
- Kulkarni 论文原始架构已在 SEMNET 实验中验证

---

### A23. GBDT + NN 加权集成 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Yan et al., "Team up GBDTs and DNNs: Tree-hybrid MLPs" (T-MLP) | KDD 2024 (Oral) | 浙江大学; U. Notre Dame | A* | GBDT 特征门控 + MLP 协同训练，88 数据集验证混合策略有效 |
| Grinsztajn et al., NeurIPS 2022 | 同 A21 | — | A* | GBDT 和 NN 各有擅长场景，互补集成有理论支撑 |

**结论**：完全可行。KDD 2024 Oral 直接验证 GBDT+NN 混合策略。

---

## 七、融合层（S6）

### A24. Meta-learning Stacking (RF 元学习器) ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Ghasemian et al., "Stacking models for nearly optimal link prediction in complex networks" | PNAS 2020 | Colorado Boulder; Harvard; USC; Santa Fe | A* | 203 算法 × 550 网络：无单一算法普适，Stacking 接近最优 |
| Wang et al., "Link Prediction Using RFE and Stacking Ensemble Learning" | Entropy 2022 | 华北电力大学 | SCI Q2 | RF 递归特征消除 + 两级 Stacking 在链接预测表现优异 |

**结论**：完全可行。PNAS 直接验证 Stacking 用于链接预测融合的优越性。

---

### A25. 概念→论文跨粒度映射 ⚠️

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Gu & Krenn, "Forecasting high-impact research topics via ML on evolving knowledge graphs" | ML: Science and Technology 2025 | Max Planck Institute | A | 2100 万篇论文，概念级预测映射到论文级影响力 |

**结论**：可行但创新性高。仅 1 篇直接论文支撑跨粒度映射方法论。该能力是本项目的新颖贡献之一，实现时需重点验证。

---

## 八、输出层（S7-S9）

### A26. 三色突变标记（新兴/成长/衰退） ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Xu et al., "Multidimensional Scientometric Indicators for Detection of Emerging Research Topics" | TFSC 2021 | 山东理工大学; CWTS Leiden | SCI Q1 (IF 12.9) | 首次提出五维新兴主题检测框架（新颖性/增长/持续性/影响力/不确定性） |
| Yang et al., "Deep Learning-based Method for Predicting Emerging Degree Using Emerging Index" | Scientometrics 2024 | 浙江财经大学 | SCI Q1/Q2 | 基于"新兴指数"量化研究主题新兴程度，异构网络 + 深度学习预测 |

**结论**：完全可行。新兴主题检测和生命周期分类是科学计量学成熟方向。

---

### A27. UMAP-3D 降维可视化 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| McInnes et al., "UMAP: Uniform Manifold Approximation and Projection" | JOSS 2018 | Tutte Institute | 方法论原始论文 | 引用 10000+，保留全局结构优于 t-SNE，支持任意维度（含 3D） |
| Becht et al., "Dimensionality Reduction for Visualizing Single-cell Data Using UMAP" | Nature Biotechnology 2019 | Fred Hutchinson; A*STAR | IF ~60 顶刊 | 大规模高维数据中最快、最可重复、最有意义的聚类组织 |

**结论**：完全可行。UMAP 是当前降维可视化标准方法。

---

### A28. 时序 Hold-out 评估策略 ✅

| 论文 | 期刊/会议 | 机构 | 等级 | 核心发现 |
|------|----------|------|------|---------|
| Huang et al., "Temporal Graph Benchmark for Machine Learning on Temporal Graphs" (TGB) | NeurIPS 2023 | McGill/Mila; Stanford; Imperial College | A* | 随机划分导致性能虚高 (>95% AP)，时序严格划分必不可少 |
| Jiao et al., "Impacts of Data Splitting Strategies on Link Prediction Algorithms" | Physica A 2026 | — | SCI | 信息泄漏平均导致 3.6% 性能高估，特定算法偏差超 15% |

**结论**：完全可行且必要。NeurIPS 2023 TGB 直接证明时序 hold-out 是链接预测评估的必备条件。

---

## 汇总矩阵

| # | 原子能力 | 所属阶段 | 验证状态 | 最高论文等级 | 风险等级 |
|---|---------|---------|---------|------------|---------|
| A1 | OpenAlex API 数据爬取 | S0 | ✅ 强 | JCR Q1 × 3 | 低 |
| A2 | Pydantic v2 + ULID | S1 | 🔧 工具级 | N/A | 低 |
| A3 | all-mpnet-base-v2 嵌入 | S1 | ⚠️ 间接 | CCF-B | 低（可升级 SPECTER2） |
| A4 | SPC Main Path | S2 | ✅ 强 | A* (JASIST) | 低 |
| A5 | BC + Co-citation | S2 | ✅ 强 | A* (JASIST) | 低 |
| A6 | KeystoneScore | S3 | ✅ 强 | JCR Q1 × 3 | 低 |
| A7 | SciBERT 引用分类 | S5a | ✅ 强 | A* (NAACL/TACL) | 低 |
| A8 | VGAE 链接预测 | S5b | ✅ 强 | CIKM 2025 最佳论文 | 低 |
| A9 | D-VGAE 升级 | Phase 3 | ✅ 直接 | A* (WWW 2024) | 中（增量价值待验证） |
| A10 | Limitation Tracking | S5c | ✅ 强 | A* (ACL/EMNLP 2025) | 低 |
| A11 | LLM 概念提取 | S5d | ✅ 强 | Nature Comms 2024 | 低 |
| A12 | KeyBERT + MMR | S5d | ✅ 强 | CORE A (CoNLL) | 低 |
| A13 | 嵌入余弦去重 | S5d | ✅ 强 | CCF-A (PVLDB) | 低 |
| A14 | 嵌入聚类归一化 | S5d | ✅ 强 | CORE A (NAACL) | 低（阈值需调优） |
| A15 | spaCy lemmatization | S5d | 🔧 工具级 | N/A | 低 |
| A16 | 概念共现网络 | S5e | ✅ 强 | SSCI Q1 + SCI Q1 | 低 |
| A17 | 拓扑特征工程 | S5e | ✅ 强 | PNAS Nexus 2024 | 低 |
| A18 | Leiden 社区检测 | S5e | ✅ 强 | SCI Q1 (Nature) | 低 |
| A19 | TransE/DistMult KGE | S5e | ✅ 强 | CCF-B (EMNLP) + SCI Q1 | 低 |
| A20 | SHAP 可解释性 | S5e | ⚠️ 间接 | CCF-A (NeurIPS/WWW) | 低（工具成熟） |
| A21 | XGBoost + LightGBM | S5e | ✅ 强 | A* (NeurIPS × 2) | 低 |
| A22 | 前馈 NN 链接预测 | S5e | ⚠️ 间接 | A* (间接) | 低（Kulkarni 已验证） |
| A23 | GBDT + NN 集成 | S5e | ✅ 强 | A* (KDD 2024 Oral) | 低 |
| A24 | Meta-learning Stacking | S6 | ✅ 强 | A* (PNAS 2020) | 低 |
| A25 | 概念→论文跨粒度映射 | S6 | ⚠️ 部分 | A (ML:ST 2025) | 中（创新性高） |
| A26 | 三色突变标记 | S7 | ✅ 强 | SCI Q1 (TFSC) | 低 |
| A27 | UMAP-3D 可视化 | S8 | ✅ 强 | Nature Biotech (IF ~60) | 低 |
| A28 | 时序 Hold-out 评估 | 评估 | ✅ 强 | A* (NeurIPS 2023) | 低 |

---

## 关键发现

### 1. 全部 28 项原子能力均可行

没有任何一项能力缺乏技术可行性支撑。20 项有强论文直接验证，5 项有间接/部分验证，3 项属于成熟工具无需论文。

### 2. 需要重点关注的 3 个能力

| 能力 | 风险 | 缓解措施 |
|------|------|---------|
| A3 all-mpnet-base-v2 | 通用模型在学术文本上可能不如专用模型 | Phase 2 评估 SPECTER2 升级 |
| A9 D-VGAE | CIKM 2025 证明调优标准 GAE 也能 SOTA | Phase 3 升级前需在项目数据上验证增量价值 |
| A25 概念→论文映射 | 创新性高，仅 1 篇直接论文 | 实现时重点做消融实验验证映射质量 |

### 3. 论文来源分布

引用论文共计 **52 篇正式发表论文**，来源分布：
- A* 级会议/期刊：~20 篇（NeurIPS, ACL, EMNLP, WWW, KDD, PNAS, Nature 系列等）
- A 级会议/期刊：~15 篇（WSDM, CIKM, JASIST, J. Informetrics 等）
- Q1/Q2 期刊：~17 篇（Scientometrics, TFSC, Neurocomputing 等）

### 4. 参数调优清单

以下参数在论文中验证了方法论合理性，但具体数值需在项目数据（光学+AI 论文）上标定：

| 参数 | 当前值 | 调优方式 |
|------|--------|---------|
| MMR diversity | 0.7 | 网格搜索 [0.5, 0.6, 0.7, 0.8] |
| 概念去重余弦阈值 | 0.90 | 人工抽检 + F1 优化 |
| 同义词聚类 cosine distance | 0.15 | 层次聚类 dendrogram 分析 |
| 时间衰减 λ | 0.8 | 网格搜索 [0.6, 0.7, 0.8, 0.9] |
| GBDT:NN 集成权重 | 0.6:0.4 | 交叉验证最优权重搜索 |
