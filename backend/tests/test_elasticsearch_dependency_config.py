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
    assert "9200:9200" in service["ports"]
    assert "discovery.type=single-node" in service["environment"]
    assert "xpack.security.enabled=false" in service["environment"]
    assert "ES_JAVA_OPTS=-Xms1g -Xmx1g" in service["environment"]
    assert "nexuskb-elasticsearch-data:/usr/share/elasticsearch/data" in service["volumes"]
    assert service["healthcheck"]["test"]
    assert "nexuskb-elasticsearch-data" in compose["volumes"]
