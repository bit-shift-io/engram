import pytest

from engram.config import Config, load_config


def write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return p


def test_loads_yaml_into_dataclass(tmp_path):
    cfg = load_config(write(tmp_path, "model_name: m\ndevice: cuda\nk: 8\n"))
    assert isinstance(cfg, Config)
    assert (cfg.model_name, cfg.device, cfg.k, cfg.chunk_size) == ("m", "cuda", 8, 128)


def test_override_field(tmp_path):
    cfg = load_config(write(tmp_path, "k: 8\n"), overrides={"k": 4})
    assert cfg.k == 4


def test_rejects_unknown_field(tmp_path):
    with pytest.raises(ValueError, match="bogus"):
        load_config(write(tmp_path, "bogus: 1\n"))


def test_default_config_file_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.device == "cpu"
