"""
Neo4j Writer — 三元组写入 Neo4j + Cypher 语句生成。

用 MERGE（而非 CREATE）写入三元组，先查重再建节点，避免同一实体重复建节点
把图谱变成一团乱麻。neo4j 驱动延迟导入——未安装 / 未启动 Neo4j 时不影响
抽取与 Cypher 生成，只有真正 write 时才要求驱动可用。

Cypher 语句与写入分离：``build_cypher`` 是纯函数（无需驱动即可测试），
``write_triples`` 才真正连接 Neo4j 执行。

使用示例:
    from src.knowledge_graph.neo4j_writer import Neo4jWriter

    writer = Neo4jWriter(uri="bolt://localhost:7687", user="neo4j", password="password")
    n = writer.write_triples(triples)
    writer.close()
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _label(entity_type: str) -> str:
    """把实体类型转成 Neo4j 标签（首字母大写，仅保留字母数字）。"""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", entity_type)
    return cleaned[:1].upper() + cleaned[1:].lower() if cleaned else "Entity"


def _quote(name: str) -> str:
    """把实体名安全地内联进 Cypher 单引号字符串。"""
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


class Neo4jWriter:
    """把三元组写入 Neo4j 图数据库。

    Attributes:
        uri:      Neo4j bolt 地址
        user:     用户名
        password: 密码
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        driver: Any = None,
    ):
        """初始化 Writer。

        Args:
            uri:      bolt 地址
            user:     用户名
            password: 密码
            driver:   注入的 Neo4j driver（测试用，绕过真实连接）
        """
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = driver

    # ── 公开接口 ────────────────────────────────────────────────────────

    @staticmethod
    def build_cypher(triples: list[dict[str, str]]) -> list[str]:
        """把三元组转成 MERGE Cypher 语句（纯函数，无需驱动）。

        Args:
            triples: 三元组列表

        Returns:
            List[str] — 每个三元组一条 MERGE 语句
        """
        statements: list[str] = []
        for t in triples:
            head_label = _label(t.get("head_type", "Entity"))
            head_name = _quote(t.get("head_name", ""))
            tail_label = _label(t.get("tail_type", "Entity"))
            tail_name = _quote(t.get("tail_name", ""))
            relation = re.sub(r"[^A-Z_]", "", t.get("relation", "RELATED_TO").upper()) or "RELATED_TO"
            statements.append(
                f"MERGE (a:{head_label} {{name: {head_name}}}) "
                f"MERGE (b:{tail_label} {{name: {tail_name}}}) "
                f"MERGE (a)-[:{relation}]->(b)"
            )
        return statements

    def write_triples(self, triples: list[dict[str, str]]) -> int:
        """把三元组写入 Neo4j，返回写入的语句数。

        Args:
            triples: 三元组列表

        Returns:
            int — 执行的 MERGE 语句数
        """
        statements = self.build_cypher(triples)
        if not statements:
            return 0

        driver = self._get_driver()
        with driver.session() as session:
            for stmt in statements:
                session.run(stmt)
        logger.info("Written %d triples to Neo4j", len(statements))
        return len(statements)

    def close(self) -> None:
        """关闭 Neo4j 连接（若已建立）。"""
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception as e:
                logger.warning("Neo4j close failed: %s", e)

    # ── 内部实现 ────────────────────────────────────────────────────────

    def _get_driver(self):
        """获取 Neo4j driver（延迟导入，未安装时给出明确提示）。"""
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import GraphDatabase  # 延迟导入
        except ImportError as e:
            raise RuntimeError(
                "neo4j 驱动未安装，请先 `uv add neo4j` 并启动 Neo4j（docker-compose up neo4j）"
            ) from e

        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        return self._driver
