from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DEV_PORT = "5173"


def test_vite_dev_server_uses_windows_safe_port():
    vite_config = (REPO_ROOT / "front" / "vite.config.js").read_text(encoding="utf-8")

    assert f"port: {FRONTEND_DEV_PORT}" in vite_config
    assert "port: 3000" not in vite_config


def test_start_dev_references_vite_dev_server_port():
    start_dev = (REPO_ROOT / "start-dev.ps1").read_text(encoding="utf-8")

    assert f"Assert-PortAvailable -Port {FRONTEND_DEV_PORT}" in start_dev
    assert f"NexusKB Frontend :{FRONTEND_DEV_PORT}" in start_dev
    assert f"Frontend: http://127.0.0.1:{FRONTEND_DEV_PORT}" in start_dev
    assert "Assert-PortAvailable -Port 3000" not in start_dev
