"""
Docker 配置 — 单元测试。

运行: uv run pytest tests/test_docker_config.py -v

用 yaml.safe_load 解析 docker-compose.yml，验证服务编排、端口、
环境变量与依赖关系，不实际启动容器。
"""

from pathlib import Path

import pytest
import yaml

COMPOSE_PATH = Path(__file__).parent.parent / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose():
    if not COMPOSE_PATH.exists():
        pytest.skip("docker-compose.yml not found")
    with open(COMPOSE_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestComposeServices:
    """测试三服务编排。"""

    def test_three_services(self, compose):
        services = compose["services"]
        assert {"app", "chroma", "neo4j"}.issubset(set(services))

    def test_app_build_and_port(self, compose):
        app = compose["services"]["app"]
        assert app["build"] == "."
        assert "8501:8501" in app["ports"]

    def test_app_depends_on(self, compose):
        app = compose["services"]["app"]
        assert "chroma" in app["depends_on"]
        assert "neo4j" in app["depends_on"]

    def test_app_env_file(self, compose):
        app = compose["services"]["app"]
        assert app["env_file"] == [".env"]


class TestComposeDependencies:
    """测试 chroma / neo4j 服务配置。"""

    def test_neo4j_ports(self, compose):
        neo4j = compose["services"]["neo4j"]
        assert "7474:7474" in neo4j["ports"]
        assert "7687:7687" in neo4j["ports"]

    def test_neo4j_auth(self, compose):
        neo4j = compose["services"]["neo4j"]
        assert "NEO4J_AUTH=neo4j/password" in neo4j["environment"]

    def test_chroma_port(self, compose):
        chroma = compose["services"]["chroma"]
        assert "8000:8000" in chroma["ports"]


class TestComposeVolumes:
    """测试数据卷持久化。"""

    def test_volumes_defined(self, compose):
        assert "chroma_data" in compose["volumes"]
        assert "neo4j_data" in compose["volumes"]
