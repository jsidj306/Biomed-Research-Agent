"""
检索评测集 — 带标注的生物医学检索 benchmark（第3-4周检查项28）。

用途：
    用「人工标注相关文档」的查询集实测混合检索的 MRR / Recall@10，
    验证是否达到第3-4周硬性验收线（MRR > 0.70，Recall@10 > 0.85）。

组成：
    - CORPUS：60 篇生物医学文献（title + abstract），覆盖 10 个主题，每个主题 6 篇。
    - QUERIES：40 条自然语言查询，每条标注了相关文档 ID 集合（人工 ground truth）。
    - TfidfSemanticRanker：一个无外部依赖的向量空间语义排序器（TF-IDF 余弦），
      作为「语义检索」路的参考实现，与 BM25 稀疏路做 RRF 融合。

说明：
    生产环境的语义路是 PubMedBERT 嵌入 + ChromaDB；TF-IDF 是弱化版的语义信号，
    因此用本评测集测出的数字是「保守下界」——PubMedBERT 只会更好，不会更差。

使用示例：
    from src.rag.eval_set import CORPUS, build_eval_set, TfidfSemanticRanker
    from src.rag.hybrid_search import HybridRetriever, mean_reciprocal_rank, recall_at_k

    corpus = [f"{d['title']}. {d['abstract']}" for d in CORPUS]
    ranker = TfidfSemanticRanker(corpus)
    retriever = HybridRetriever(corpus)
    eval_set = build_eval_set()
    mrr = mean_reciprocal_rank(eval_set, lambda q: [r["index"] for r in retriever.search(q, ranker.rank, k=10)])
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from src.rag.hybrid_search import _tokenize

# ── 语料 + 查询（按主题组织，便于人工审阅相关性标注）──────────────────────────


TOPICS: dict[str, dict[str, Any]] = {
    "car_t_cd19": {
        "docs": [
            {
                "title": "Chimeric antigen receptor T cells targeting CD19 in refractory B-cell acute lymphoblastic leukemia",
                "abstract": "Autologous T cells engineered to express a CD19-specific chimeric antigen receptor induced durable remission in patients with refractory B-cell acute lymphoblastic leukemia.",
            },
            {
                "title": "Anti-CD19 CAR T-cell therapy for diffuse large B-cell lymphoma",
                "abstract": "CD19-directed CAR T cells produced high response rates in relapsed diffuse large B-cell lymphoma, though cytokine release syndrome remains a common toxicity.",
            },
            {
                "title": "Cytokine release syndrome following CAR T-cell infusion",
                "abstract": "Cytokine release syndrome is a frequent and potentially life-threatening complication of CAR T-cell therapy, driven by massive T-cell activation and release of inflammatory cytokines.",
            },
            {
                "title": "Overcoming barriers to CAR T-cell therapy in solid tumors",
                "abstract": "Unlike hematologic malignancies, solid tumors resist CAR T-cell therapy because of antigen heterogeneity, an immunosuppressive microenvironment, and poor T-cell trafficking.",
            },
            {
                "title": "CD19 antigen loss as a mechanism of CAR T-cell resistance",
                "abstract": "Relapse after anti-CD19 CAR T-cell therapy is frequently associated with loss of CD19 expression on malignant B cells, an immune escape mechanism.",
            },
            {
                "title": "Engineering next-generation CARs with dual costimulation",
                "abstract": "Second and third generation CAR constructs incorporate costimulatory domains such as CD28 and 4-1BB to enhance T-cell persistence and effector function.",
            },
        ],
        "queries": [
            {"q": "How effective are CAR-T cells against CD19-positive blood cancers?", "rel": [0, 1]},
            {"q": "What serious side effects can CAR T-cell therapy cause?", "rel": [2]},
            {"q": "Why does CAR-T therapy struggle to treat solid tumors?", "rel": [3]},
            {"q": "How do leukemia cells escape CAR-T targeting?", "rel": [4]},
        ],
    },
    "mrna_covid": {
        "docs": [
            {
                "title": "Lipid nanoparticle delivery of mRNA vaccines against SARS-CoV-2",
                "abstract": "mRNA vaccines encapsulate the spike protein coding sequence inside lipid nanoparticles, which protect the RNA and enable intracellular delivery to antigen-presenting cells.",
            },
            {
                "title": "Efficacy of the mRNA-1273 vaccine against symptomatic COVID-19",
                "abstract": "The mRNA-1273 vaccine demonstrated roughly 94 percent efficacy against symptomatic COVID-19 in a large randomized placebo-controlled trial.",
            },
            {
                "title": "mRNA vaccine platform from cancer immunotherapy to infectious disease",
                "abstract": "The same mRNA platform originally developed for cancer immunotherapy was rapidly repurposed to encode viral antigens, enabling fast vaccine design.",
            },
            {
                "title": "Rare adverse events after mRNA COVID-19 vaccination",
                "abstract": "mRNA COVID-19 vaccines are associated with rare adverse events including myocarditis in younger males, which must be weighed against the risks of infection.",
            },
            {
                "title": "Booster doses and waning immunity after mRNA vaccination",
                "abstract": "Neutralizing antibody titers decline over months after primary mRNA vaccination, and booster doses restore protection against severe disease.",
            },
            {
                "title": "Stability and cold-chain requirements of mRNA vaccines",
                "abstract": "mRNA vaccines are inherently unstable and require ultra-cold storage, prompting research into lyophilized and thermostable formulations.",
            },
        ],
        "queries": [
            {"q": "How are mRNA vaccines delivered into cells?", "rel": [0]},
            {"q": "How well does the mRNA-1273 vaccine prevent COVID-19?", "rel": [1]},
            {"q": "What side effects do mRNA COVID vaccines cause?", "rel": [3]},
            {"q": "Do mRNA vaccine antibodies wane over time?", "rel": [4]},
        ],
    },
    "crispr": {
        "docs": [
            {
                "title": "CRISPR-Cas9 genome editing in mammalian cells",
                "abstract": "The CRISPR-Cas9 system uses a guide RNA to direct the Cas9 nuclease to a complementary DNA sequence, producing a targeted double-strand break that enables gene editing.",
            },
            {
                "title": "Off-target effects of CRISPR-Cas9",
                "abstract": "Cas9 can cleave sequences that partially match the guide RNA, causing unintended off-target mutations that must be detected and minimized for safe editing.",
            },
            {
                "title": "Delivery of CRISPR components using adeno-associated virus",
                "abstract": "Adeno-associated virus vectors can deliver Cas9 and guide RNA to target tissues in vivo, though packaging limits and immunogenicity remain challenges.",
            },
            {
                "title": "CRISPR base editing for single-nucleotide correction",
                "abstract": "Base editors fuse a catalytically impaired Cas9 to a deaminase enzyme to convert one DNA base into another without creating a double-strand break.",
            },
            {
                "title": "Ethical considerations in human germline editing",
                "abstract": "Editing human embryos raises profound ethical questions about heritable genetic modification, consent, and equity, prompting calls for international governance.",
            },
            {
                "title": "CRISPR screens for functional genomics",
                "abstract": "Pooled CRISPR knockout and activation screens systematically perturb thousands of genes in parallel to reveal their roles in biological processes.",
            },
        ],
        "queries": [
            {"q": "How does CRISPR-Cas9 edit the genome?", "rel": [0]},
            {"q": "What are the risks of off-target CRISPR edits?", "rel": [1]},
            {"q": "How can CRISPR components be delivered in vivo?", "rel": [2]},
            {"q": "What is base editing used for?", "rel": [3]},
        ],
    },
    "alzheimers": {
        "docs": [
            {
                "title": "Amyloid-beta plaques in Alzheimer's disease pathology",
                "abstract": "Extracellular accumulation of amyloid-beta peptides into plaques is a defining pathological hallmark of Alzheimer's disease and a focus of diagnostic and therapeutic targeting.",
            },
            {
                "title": "Tau protein hyperphosphorylation and neurofibrillary tangles",
                "abstract": "Hyperphosphorylated tau protein aggregates into intracellular neurofibrillary tangles, which correlate more closely with cognitive decline than amyloid plaques.",
            },
            {
                "title": "Anti-amyloid monoclonal antibodies for Alzheimer's therapy",
                "abstract": "Monoclonal antibodies that clear amyloid-beta plaques have shown modest slowing of cognitive decline in early Alzheimer's disease trials.",
            },
            {
                "title": "APOE4 as a genetic risk factor for Alzheimer's disease",
                "abstract": "The apolipoprotein E epsilon4 allele is the strongest common genetic risk factor for late-onset Alzheimer's disease, influencing amyloid clearance and neuroinflammation.",
            },
            {
                "title": "Biomarkers for early diagnosis of Alzheimer's disease",
                "abstract": "Cerebrospinal fluid and blood biomarkers such as phosphorylated tau and amyloid ratio enable detection of Alzheimer's pathology years before clinical symptoms.",
            },
            {
                "title": "Neuroinflammation in Alzheimer's disease progression",
                "abstract": "Chronic activation of microglia and astrocytes contributes to synapse loss and neurodegeneration, positioning neuroinflammation as a therapeutic target.",
            },
        ],
        "queries": [
            {"q": "What role do amyloid plaques play in Alzheimer's disease?", "rel": [0]},
            {"q": "Which antibodies target amyloid in Alzheimer's therapy?", "rel": [2]},
            {"q": "What genetic factor raises Alzheimer's risk?", "rel": [3]},
            {"q": "What biomarkers help diagnose Alzheimer's early?", "rel": [4]},
        ],
    },
    "diabetes_glp1": {
        "docs": [
            {
                "title": "GLP-1 receptor agonists for glycemic control in type 2 diabetes",
                "abstract": "Glucagon-like peptide-1 receptor agonists enhance glucose-dependent insulin secretion and suppress glucagon, improving glycemic control in type 2 diabetes.",
            },
            {
                "title": "Insulin resistance and beta-cell dysfunction in type 2 diabetes",
                "abstract": "Type 2 diabetes arises from peripheral insulin resistance combined with progressive pancreatic beta-cell dysfunction and relative insulin deficiency.",
            },
            {
                "title": "Cardiovascular benefits of GLP-1 receptor agonists",
                "abstract": "Several GLP-1 receptor agonists have demonstrated reductions in major adverse cardiovascular events in patients with type 2 diabetes at high cardiovascular risk.",
            },
            {
                "title": "SGLT2 inhibitors and renal outcomes in diabetes",
                "abstract": "Sodium-glucose cotransporter-2 inhibitors lower glucose and reduce the progression of diabetic kidney disease through hemodynamic and metabolic effects.",
            },
            {
                "title": "Metformin mechanism of action in type 2 diabetes",
                "abstract": "Metformin reduces hepatic glucose production and improves insulin sensitivity, remaining first-line therapy for type 2 diabetes.",
            },
            {
                "title": "Weight loss effects of semaglutide",
                "abstract": "Semaglutide, a long-acting GLP-1 receptor agonist, produces substantial and sustained weight loss in patients with obesity and overweight.",
            },
        ],
        "queries": [
            {"q": "How do GLP-1 receptor agonists control blood sugar?", "rel": [0]},
            {"q": "What causes type 2 diabetes at the cellular level?", "rel": [1]},
            {"q": "Do GLP-1 drugs protect the heart?", "rel": [2]},
            {"q": "How much weight can semaglutide help patients lose?", "rel": [5]},
        ],
    },
    "antibiotic_resistance": {
        "docs": [
            {
                "title": "Mechanisms of beta-lactam resistance in MRSA",
                "abstract": "Methicillin-resistant Staphylococcus aureus resists beta-lactam antibiotics through acquisition of the mecA gene encoding an altered penicillin-binding protein.",
            },
            {
                "title": "Emergence of carbapenem-resistant Enterobacteriaceae",
                "abstract": "Carbapenem-resistant Enterobacteriaceae produce carbapenemase enzymes that hydrolyze last-resort antibiotics, posing a severe public health threat.",
            },
            {
                "title": "Efflux pumps and multidrug resistance",
                "abstract": "Bacterial efflux pumps actively expel multiple classes of antibiotics from the cell, contributing to multidrug resistance in many pathogens.",
            },
            {
                "title": "Antibiotic stewardship programs to curb resistance",
                "abstract": "Antimicrobial stewardship programs optimize antibiotic prescribing to slow the emergence of resistance and improve patient outcomes.",
            },
            {
                "title": "Horizontal gene transfer of antibiotic resistance genes",
                "abstract": "Resistance genes spread between bacteria through plasmids and transposons, accelerating the global dissemination of antibiotic resistance.",
            },
            {
                "title": "Phage therapy as an alternative to antibiotics",
                "abstract": "Bacteriophages specifically infect and kill bacteria, offering a potential alternative for treating multidrug-resistant infections.",
            },
        ],
        "queries": [
            {"q": "How does MRSA resist beta-lactam antibiotics?", "rel": [0]},
            {"q": "What are carbapenem-resistant Enterobacteriaceae?", "rel": [1]},
            {"q": "How do bacteria acquire antibiotic resistance genes?", "rel": [4]},
            {"q": "Can bacteriophages treat resistant infections?", "rel": [5]},
        ],
    },
    "checkpoint_pd1": {
        "docs": [
            {
                "title": "PD-1 blockade in metastatic melanoma",
                "abstract": "Antibodies blocking the programmed cell death protein 1 receptor produce durable responses and improved survival in metastatic melanoma.",
            },
            {
                "title": "CTLA-4 inhibition and immune checkpoint blockade",
                "abstract": "Targeting cytotoxic T-lymphocyte antigen 4 releases a brake on T-cell activation, forming the basis of early checkpoint inhibitor therapy.",
            },
            {
                "title": "Immune-related adverse events of checkpoint inhibitors",
                "abstract": "Checkpoint inhibitors can trigger immune-related adverse events such as colitis, pneumonitis, and endocrinopathies due to loss of peripheral tolerance.",
            },
            {
                "title": "Biomarkers predicting response to PD-1 therapy",
                "abstract": "PD-L1 expression, tumor mutational burden, and immune infiltration are candidate biomarkers for selecting patients likely to respond to checkpoint blockade.",
            },
            {
                "title": "Combination of checkpoint inhibitors with chemotherapy",
                "abstract": "Combining checkpoint blockade with chemotherapy can improve response rates by enhancing tumor antigen release and immune activation.",
            },
            {
                "title": "Tumor mutational burden and immunotherapy response",
                "abstract": "High tumor mutational burden correlates with increased neoantigen load and improved response to checkpoint inhibitors across several cancers.",
            },
        ],
        "queries": [
            {"q": "Is PD-1 blockade effective in metastatic melanoma?", "rel": [0]},
            {"q": "What side effects do checkpoint inhibitors cause?", "rel": [2]},
            {"q": "Which biomarkers predict immunotherapy response?", "rel": [3, 5]},
            {"q": "Why combine checkpoint inhibitors with chemotherapy?", "rel": [4]},
        ],
    },
    "parkinsons": {
        "docs": [
            {
                "title": "Alpha-synuclein aggregation in Parkinson's disease",
                "abstract": "Misfolded alpha-synuclein aggregates into Lewy bodies in dopaminergic neurons, a central pathological feature of Parkinson's disease.",
            },
            {
                "title": "Deep brain stimulation for advanced Parkinson's disease",
                "abstract": "High-frequency electrical stimulation of the subthalamic nucleus improves motor symptoms in patients with advanced Parkinson's disease.",
            },
            {
                "title": "Dopaminergic neuron loss in the substantia nigra",
                "abstract": "Progressive degeneration of dopamine-producing neurons in the substantia nigra pars compacta underlies the motor symptoms of Parkinson's disease.",
            },
            {
                "title": "LRRK2 mutations and familial Parkinson's disease",
                "abstract": "Mutations in the leucine-rich repeat kinase 2 gene are the most common genetic cause of familial and sporadic Parkinson's disease.",
            },
            {
                "title": "Gut-brain axis in Parkinson's disease",
                "abstract": "Emerging evidence links gut microbiota and intestinal inflammation to the initiation of alpha-synuclein pathology and Parkinson's disease progression.",
            },
            {
                "title": "Neuroprotective strategies in Parkinson's disease",
                "abstract": "Neuroprotective approaches aim to slow dopaminergic degeneration through antioxidants, anti-inflammatory agents, and targeted molecular therapies.",
            },
        ],
        "queries": [
            {"q": "What protein misfolds in Parkinson's disease?", "rel": [0]},
            {"q": "How is deep brain stimulation used in Parkinson's?", "rel": [1]},
            {"q": "Which neurons degenerate in Parkinson's disease?", "rel": [2]},
            {"q": "What genetic mutation causes familial Parkinson's?", "rel": [3]},
        ],
    },
    "cardiovascular_statins": {
        "docs": [
            {
                "title": "Statins reduce LDL cholesterol and cardiovascular events",
                "abstract": "Statin therapy lowers low-density lipoprotein cholesterol and reduces the incidence of myocardial infarction and stroke in at-risk patients.",
            },
            {
                "title": "PCSK9 inhibitors for familial hypercholesterolemia",
                "abstract": "Monoclonal antibodies against PCSK9 dramatically lower LDL cholesterol in patients with familial hypercholesterolemia who are intolerant of statins.",
            },
            {
                "title": "Atherosclerosis and the role of inflammation",
                "abstract": "Atherosclerosis is a chronic inflammatory disease of the arterial wall in which lipid deposition and immune cell infiltration drive plaque formation.",
            },
            {
                "title": "Blood pressure control and stroke prevention",
                "abstract": "Lowering elevated blood pressure is the single most effective intervention for preventing stroke and other cardiovascular events.",
            },
            {
                "title": "Triglycerides and residual cardiovascular risk",
                "abstract": "Elevated triglycerides contribute to residual cardiovascular risk even after LDL cholesterol is controlled with statin therapy.",
            },
            {
                "title": "Aspirin in primary prevention of cardiovascular disease",
                "abstract": "Low-dose aspirin modestly reduces cardiovascular events in primary prevention but increases bleeding risk, limiting its net benefit.",
            },
        ],
        "queries": [
            {"q": "Do statins lower the risk of heart attacks?", "rel": [0]},
            {"q": "What do PCSK9 inhibitors treat?", "rel": [1]},
            {"q": "How does inflammation drive atherosclerosis?", "rel": [2]},
            {"q": "Does aspirin prevent heart disease in healthy people?", "rel": [5]},
        ],
    },
    "aav_gene_therapy": {
        "docs": [
            {
                "title": "Adeno-associated virus vectors for retinal gene therapy",
                "abstract": "AAV vectors delivered by subretinal injection restore vision in inherited retinal dystrophies such as RPE65-associated Leber congenital amaurosis.",
            },
            {
                "title": "AAV-mediated gene therapy for spinal muscular atrophy",
                "abstract": "Intravenous AAV9 delivering the SMN1 gene improves motor function and survival in infants with spinal muscular atrophy.",
            },
            {
                "title": "Immune responses to AAV capsid",
                "abstract": "Pre-existing neutralizing antibodies against the AAV capsid and capsid-specific T-cell responses can limit transgene expression and durability.",
            },
            {
                "title": "Hemophilia B gene therapy with factor IX",
                "abstract": "AAV-based gene therapy delivering factor IX achieves sustained factor expression and reduces bleeding in hemophilia B patients.",
            },
            {
                "title": "In vivo genome editing with AAV-delivered CRISPR",
                "abstract": "AAV vectors can co-deliver Cas9 and guide RNA to perform in vivo genome editing in the liver and other organs.",
            },
            {
                "title": "Manufacturing challenges for AAV gene therapy vectors",
                "abstract": "Producing high-titer, high-purity AAV vectors at scale remains a major cost and quality bottleneck for gene therapy.",
            },
        ],
        "queries": [
            {"q": "What retinal diseases can AAV gene therapy treat?", "rel": [0]},
            {"q": "How does gene therapy treat spinal muscular atrophy?", "rel": [1]},
            {"q": "What limits AAV gene therapy in hemophilia?", "rel": [3]},
            {"q": "Can AAV deliver CRISPR for in vivo editing?", "rel": [4]},
        ],
    },
}

# ── 展平成全局索引（供检索器直接使用）────────────────────────────────────────

CORPUS: list[dict[str, Any]] = []
QUERIES: list[dict[str, Any]] = []

for _topic in TOPICS.values():
    _base = len(CORPUS)
    for _doc in _topic["docs"]:
        CORPUS.append({"id": len(CORPUS), "title": _doc["title"], "abstract": _doc["abstract"]})
    for _q in _topic["queries"]:
        QUERIES.append(
            {"query": _q["q"], "relevant_ids": [_base + i for i in _q["rel"]]}
        )


def build_eval_set() -> list[tuple[str, set[int]]]:
    """返回标准评测集格式 [(query, 相关文档ID集合), ...]。

    Returns:
        List[(str, set[int])] — 供 mean_reciprocal_rank / recall_at_k 直接消费。
    """
    return [(q["query"], set(q["relevant_ids"])) for q in QUERIES]


# ── TF-IDF 语义排序器（评测用参考实现，无外部依赖）──────────────────────────


class TfidfSemanticRanker:
    """向量空间模型语义排序器（TF-IDF 余弦相似度）。

    作为「语义检索」路的确定性参考实现：对 query 和每篇文档各算一个
    TF-IDF 加权向量，按余弦相似度降序返回文档下标。它捕获的是「软词频
    加权」语义，弱于生产环境的 PubMedBERT 嵌入，因此是保守下界。

    Attributes:
        documents: 文档文本列表
    """

    def __init__(self, documents: list[str]):
        """构建 TF-IDF 索引。

        Args:
            documents: 文档文本列表
        """
        self.documents = list(documents)
        self._tokens = [_tokenize(d) for d in self.documents]
        self._n = len(self._tokens)

        self._df: Counter = Counter()
        for tokens in self._tokens:
            for term in set(tokens):
                self._df[term] += 1

        self._doc_vec = [self._vector(tokens) for tokens in self._tokens]

    def _idf(self, term: str) -> float:
        return math.log((self._n + 1) / (self._df.get(term, 0) + 1)) + 1.0

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        """TF-IDF 向量，L2 归一化。"""
        tf = Counter(tokens)
        vec = {t: c * self._idf(t) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0:
            return {}
        return {t: v / norm for t, v in vec.items()}

    def rank(self, query: str, k: int | None = None) -> list[int]:
        """按余弦相似度降序返回文档下标。

        Args:
            query: 查询文本
            k:     返回结果数（None 表示全部）

        Returns:
            List[int] — 文档下标，按相似度降序
        """
        qv = self._vector(_tokenize(query))
        if not qv:
            return []

        scored: list[tuple[int, float]] = []
        for i, dv in enumerate(self._doc_vec):
            if not dv:
                continue
            dot = sum(qv.get(term, 0.0) * w for term, w in dv.items())
            if dot > 0:
                scored.append((i, dot))
        scored.sort(key=lambda x: x[1], reverse=True)
        idxs = [i for i, _ in scored]
        return idxs if k is None else idxs[:k]
