# 用 LLM + Neo4j 给生物医学文献建知识图谱

> 这是「Biomed-Research-Agent」项目的第三篇技术复盘。
> 上一篇结尾我说，一个研究助手光会「挑错」还不够，还得能把读过的论文**组织成可推理的知识结构**。
> 这一篇就讲这件事：用 LLM 抽取 + Neo4j 存储，给生物医学文献建一张知识图谱。

---

## 一、为什么生物医学适合知识图谱

先想一个问题：生物医学的知识，到底长什么样？

不是表格，不是列表，是**图**。药物、靶点、疾病、基因之间天然是一张网：

```
CAR-T ──TREATS──▶ 白血病
  │
  └──TARGETS──▶ CD19 ──ASSOCIATED_WITH──▶ 白血病
```

一条「CAR-T 靶向 CD19、治疗 B 细胞白血病」的知识，在表格里要拆成几行才能存下，在图里就是三个节点、两条边，一眼看穿。而「检索文献」和「组织知识」是两回事：前者给你一堆论文，后者给你一张能推理的网络——这正是知识图谱的价值。

所以第三周，我给 Agent 加了一条新管线：**从论文摘要里抽取「药物-靶点-疾病」三元组，写进 Neo4j**。

但动手前我就知道难点不在「怎么建图」，而在三件事：

1. **让 LLM 抽取可靠**——LLM 会脑补，抽出一堆文本里根本不存在的实体关系；
2. **让写入幂等**——同一批数据重复写，不能把图谱刷成一团乱麻；
3. **离数据库也能测**——单测不能依赖一个真的 Neo4j 实例。

下面按这三件事展开。

---

## 二、抽取：LLM 主抽取 + 词典兜底

### 2.1 先限死实体和关系，再让 LLM 抽

这是最关键的一步。如果不限定，LLM 会抽出「论文提到 X」「作者是 Y」这种无意义的关系——它在自由发挥时，什么都能给你连起来。

所以我先画死了边界：

```python
ENTITY_TYPES = ("drug", "target", "disease", "gene")

# 允许的关系（head → tail）
RELATIONS = {
    "TREATS": ("drug", "disease"),           # 药物 治疗 疾病
    "TARGETS": ("drug", "target"),           # 药物 靶向 靶点
    "ASSOCIATED_WITH": ("target", "disease"), # 靶点 关联 疾病
    "ENCODES": ("gene", "target"),           # 基因 编码 靶点
    "IMPLICATED_IN": ("gene", "disease"),    # 基因 涉及 疾病
}
```

注意 `RELATIONS` 不仅限定了关系名，还限定了**头尾实体类型**（`TREATS` 只能是 drug→disease，不能是 disease→drug）。这相当于给 LLM 套了一个 schema 的紧箍咒——它只能在这个词汇表里发挥。

prompt 里也反复强调两件事：**只抽文本里明确出现的实体**、**没有就别硬凑**：

```python
KG_PROMPT = """你是生物医学知识图谱抽取专家。从下面的文本中抽取实体关系三元组。
实体类型限定为: drug / target / disease / gene。
关系限定为: TREATS / TARGETS / ASSOCIATED_WITH / ENCODES / IMPLICATED_IN。
只抽取文本中明确出现的实体，不要臆造。若无实体关系，返回 {"triples": []}。"""
```

### 2.2 词典兜底：LLM 抽不到的用规则补

LLM 会漏，也可能因为限流/网络挂了直接失败。所以我配了一个**纯规则的词典兜底**——预置常见的药物/靶点/疾病词典，做小写子串匹配：

```python
DRUG_DICT = ("car-t", "rituximab", "pembrolizumab", "imatinib", "aspirin", "metformin")
TARGET_DICT = ("cd19", "pd-1", "pd-l1", "bcr-abl", "egfr", "her2")
DISEASE_DICT = ("cancer", "leukemia", "lymphoma", "melanoma", "diabetes", "alzheimer", "tumor", "carcinoma")
```

兜底有个**必须守住的约束**：只对「同一文本中**同时出现**」的实体对生成三元组。

```python
lower = text.lower()
drugs = [d for d in DRUG_DICT if d in lower]
targets = [t for t in TARGET_DICT if t in lower]
diseases = [d for d in DISEASE_DICT if d in lower]

for drug in drugs:
    for disease in diseases:
        triples.append(_triple("drug", drug, "TREATS", "disease", disease))
# ... 同理生成 TARGETS / ASSOCIATED_WITH
```

这一条直接砍掉了「臆造」的可能：如果一句话里既没出现 `car-t` 也没出现 `leukemia`，那词典绝不会凭空造出 `car-t TREATS leukemia` 这条边。这是规则抽取和 LLM 抽取最大的区别——规则只会「没抽到」，不会「抽错」。

### 2.3 归一化 + 去重

LLM 返回的 JSON 是不可信的：实体类型可能拼错、关系可能带空格、字段名可能不对。所以每一刀都要过一遍 `_normalize`：

```python
@staticmethod
def _normalize(triple):
    head_type = str(triple.get("head", triple.get("head_type", ""))).lower()
    relation = str(triple.get("relation", "")).upper()
    # 实体类型与关系合法性校验
    if head_type not in ENTITY_TYPES or tail_type not in ENTITY_TYPES:
        return None
    if relation not in RELATIONS:
        return None
    if not head_name or not tail_name:
        return None
    return _triple(head_type, head_name, relation, tail_type, tail_name)
```

非法项**直接丢弃**（返回 `None` 过滤掉），而不是「尽力修正」——宁可少抽，不能抽错。最后 LLM 抽取 + 词典兜底两条来源合并，按 `(head_type, head_name, relation, tail_type, tail_name)` 五元组去重：

```python
merged = self._dedupe(llm_triples + dict_triples)
```

---

## 三、写入：为什么用 MERGE 而不是 CREATE

这是我最想讲清楚的一点。往图数据库里写数据，新手最容易写错的就是用 `CREATE`。

`CREATE` 的语义是「无条件新建」——跑一次建一个节点，跑十次建十个重复节点。同一批论文你重复跑几次抽取，图谱里就会堆满 `CAR-T`、`CAR-T`、`CAR-T`……变成一团乱麻。

`MERGE` 的语义是「先查重，没有才建」。同一句 `MERGE (a:Drug {name:'CAR-T'})`，跑一百次也只有**一个** `CAR-T` 节点——这就是**幂等**：重复执行结果不变。

我的 `build_cypher` 就是纯 MERGE：

```python
def build_cypher(triples):
    statements = []
    for t in triples:
        # ...
        statements.append(
            f"MERGE (a:{head_label} {{name: {head_name}}}) "
            f"MERGE (b:{tail_label} {{name: {tail_name}}}) "
            f"MERGE (a)-[:{relation}]->(b)"
        )
    return statements
```

三段 MERGE 分别保证：头节点唯一、尾节点唯一、关系唯一。我在本地 Neo4j 实测过：12 个三元组写进去是 **6 个节点、12 条关系**（多个三元组共享同一批节点），重复写入数量不变——幂等成立。

> 一句话：**`CREATE` 是「无条件插入」，`MERGE` 是「存在即跳过」**。幂等是数据管线的底线，做不到幂等，任何重试/重跑都会变成数据事故。

---

## 四、工程解耦：Cypher 与连接分离

这是我最得意的一处工程决策：**把「生成 Cypher 语句」和「真正连数据库」拆成两个函数。**

- `build_cypher(triples)` 是**纯函数**——输入三元组，输出 Cypher 字符串，不碰任何数据库；
- `write_triples(triples)` 才真正连 Neo4j，逐条执行。

```python
# build_cypher：纯函数，无需驱动
statements = Neo4jWriter.build_cypher(triples)

# write_triples：真正连库
count = writer.write_triples(triples)
```

好处立竿见影：

1. **单元测试零依赖**。测试 `build_cypher` 只需要断言「输出字符串包含 `MERGE`、`:Drug`、`:TREATS`」，不用起一个 Neo4j 实例，CI 也不用装数据库。
2. **驱动延迟导入**。`neo4j` 这个 Python 包只在 `write_triples` 真正执行时才 import：

```python
def _get_driver(self):
    if self._driver is not None:
        return self._driver
    try:
        from neo4j import GraphDatabase  # 延迟导入
    except ImportError as e:
        raise RuntimeError("neo4j 驱动未安装，请先 `uv add neo4j` ...") from e
    self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
    return self._driver
```

也就是说，**没装 Neo4j、没启动数据库，完全不影响抽取和 Cypher 预览**——只有当用户真的想写库时才要求环境就绪。这个「能跑的部分先跑起来，缺的东西到用时才报错」的思路，和上一篇讲的「降级策略」是一脉相承的。

---

## 五、坑与心得

**1. 关系名一定要清洗。** LLM 会返回 `TREATS B-CELL`、`treats`、甚至带空格的怪关系。入库前必须强制归一：

```python
relation = re.sub(r"[^A-Z_]", "", t.get("relation", "RELATED_TO").upper()) or "RELATED_TO"
```

**2. 节点标签也要清洗。** 实体类型转标签时，首字母大写、只留字母数字（`_label` 函数），否则一个 `pd-1` 会变成非法的 Cypher 标签。

**3. 实体名内联进 Cypher 要做转义。** 实体名里可能带单引号，直接拼进字符串会破坏语法，所以 `_quote` 先把 `\` 和 `'` 转义掉再包进单引号。

**4. 词典兜底是「兜底」不是「主力」。** 它只能抓到词典里预置的那几十个实体，覆盖长尾还是要靠 LLM。正确的定位是：LLM 为主、词典保底、二者合并去重。LLM 挂了，至少还能靠词典抽出最核心的那几条边。

---

## 六、下一步

知识图谱解决了「把知识**组织**起来」。但一个 Agent 的价值，最终要落在「**能交付给别人用的东西**」上——把多轮研究的产物组装成一份结构化报告、导出成 PDF/Word、一键部署。这就是下一篇（收官篇）要讲的了。

---

*项目地址：[github.com/jsidj306/Biomed-Research-Agent](https://github.com/jsidj306/Biomed-Research-Agent)*
*2026-08*
