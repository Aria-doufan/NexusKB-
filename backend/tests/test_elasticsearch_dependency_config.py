from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_requirements_include_elasticsearch_client():
    requirements = (REPO_ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8").splitlines()

    assert "elasticsearch>=8.19.0,<9.0.0" in requirements


def test_docker_compose_elasticsearch_service_exists():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.elasticsearch.yml").read_text(encoding="utf-8"))
    service = compose["services"]["elasticsearch"]

    assert service["image"] == "docker.elastic.co/elasticsearch/elasticsearch:8.19.0"
    assert "127.0.0.1:9298:9200" in service["ports"]
    assert "discovery.type=single-node" in service["environment"]
    assert "xpack.security.enabled=false" in service["environment"]
    assert "ES_JAVA_OPTS=-Xms1g -Xmx1g" in service["environment"]
    assert "nexuskb-elasticsearch-data:/usr/share/elasticsearch/data" in service["volumes"]
    assert service["healthcheck"]["test"]
    assert "nexuskb-elasticsearch-data" in compose["volumes"]


def test_local_elasticsearch_config_uses_compose_host_port():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.elasticsearch.yml").read_text(encoding="utf-8"))
    rag_config = yaml.safe_load((REPO_ROOT / "backend" / "app" / "config" / "rag.yaml").read_text(encoding="utf-8"))
    start_dev = (REPO_ROOT / "start-dev.ps1").read_text(encoding="utf-8")
    index_script = (REPO_ROOT / "backend" / "scripts" / "index_enterprise_chunks_elasticsearch.py").read_text(encoding="utf-8")
    retrieval_backend = (REPO_ROOT / "backend" / "app" / "rag" / "retrieval_backends" / "elasticsearch_enterprise.py").read_text(encoding="utf-8")

    published_port = compose["services"]["elasticsearch"]["ports"][0].split(":")[1]

    assert rag_config["elasticsearch"]["url"] == f"http://localhost:{published_port}"
    assert f'DEFAULT_URL = "http://localhost:{published_port}"' in index_script
    assert f'url = es_config.get("url") or "http://localhost:{published_port}"' in retrieval_backend
    assert f"Test-PortListening {published_port}" in start_dev
    assert f"Wait-PortListening -Port {published_port}" in start_dev
