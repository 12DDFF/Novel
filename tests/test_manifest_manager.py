"""Tests for ManifestManager."""
import json
from pathlib import Path

import pytest

from src.core.manifest_manager import ManifestManager
from src.models import AssetStatus, Character, Manifest, Scene


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project directory."""
    return tmp_path / "test_project"


@pytest.fixture
def manager(tmp_project):
    """Create a ManifestManager for a temporary project."""
    return ManifestManager(tmp_project)


class TestManifestManager:
    def test_create_manifest(self, manager, tmp_project):
        manifest = manager.create(project_name="测试项目")
        assert manifest.project_name == "测试项目"
        assert (tmp_project / "manifest.json").exists()

    def test_exists(self, manager):
        assert not manager.exists()
        manager.create(project_name="test")
        assert manager.exists()

    def test_load(self, manager):
        manager.create(project_name="仙逆")
        loaded = manager.load()
        assert loaded.project_name == "仙逆"

    def test_save_updates_timestamp(self, manager):
        manifest = manager.create(project_name="test")
        original_time = manifest.updated_at

        import time
        time.sleep(0.01)

        manifest.project_name = "updated"
        manager.save(manifest)

        loaded = manager.load()
        assert loaded.project_name == "updated"
        assert loaded.updated_at > original_time

    def test_update_with_function(self, manager):
        manager.create(project_name="test")

        char = Character(id="char_001", name="王林")
        result = manager.update(lambda m: m.characters.append(char))

        assert len(result.characters) == 1
        assert result.characters[0].name == "王林"

        # Verify persisted
        loaded = manager.load()
        assert len(loaded.characters) == 1

    def test_update_scene_asset(self, manager):
        manifest = manager.create(project_name="test")
        scene = Scene(id="scene_001", sequence=1)
        manifest.scenes.append(scene)
        manager.save(manifest)

        manager.update_scene_asset(
            scene_id="scene_001",
            asset_type="image",
            status=AssetStatus.COMPLETE,
            path="data/images/scenes/scene_001.png",
        )

        loaded = manager.load()
        s = loaded.get_scene("scene_001")
        assert s.assets.image_status == AssetStatus.COMPLETE
        assert s.assets.image_path == "data/images/scenes/scene_001.png"

    def test_update_scene_asset_not_found(self, manager):
        manager.create(project_name="test")
        with pytest.raises(ValueError, match="not found"):
            manager.update_scene_asset("nonexistent", "image", AssetStatus.COMPLETE)

    def test_add_cost(self, manager):
        manager.create(project_name="test")
        manager.add_cost("gpu", 0.29, "RTX4090 1hr")
        manager.add_cost("llm", 0.01, "DeepSeek")

        loaded = manager.load()
        assert loaded.total_cost() == pytest.approx(0.30)
        assert len(loaded.costs) == 2

    def test_delete(self, manager):
        manager.create(project_name="test")
        assert manager.exists()
        manager.delete()
        assert not manager.exists()

    def test_roundtrip_complex_manifest(self, manager):
        """Full roundtrip with characters, scenes, costs."""
        manifest = manager.create(project_name="完美世界")
        manifest.characters = [
            Character(id="c1", name="石昊", aliases=["荒", "小不点"], role="protagonist"),
            Character(id="c2", name="石毅", role="antagonist"),
        ]
        manifest.scenes = [
            Scene(
                id="s1", sequence=1,
                narration_text="石昊站在山顶，俯瞰着整个村庄。",
                characters_present=["c1"],
            ),
            Scene(
                id="s2", sequence=2,
                narration_text="一道剑光划过天际。",
                characters_present=["c1", "c2"],
            ),
        ]
        manifest.add_cost("gpu", 0.15)
        manifest.scenes[0].assets.image_status = AssetStatus.COMPLETE
        manifest.scenes[0].assets.image_path = "img/s1.png"
        manager.save(manifest)

        # Load and verify everything survived
        loaded = manager.load()
        assert loaded.project_name == "完美世界"
        assert len(loaded.characters) == 2
        assert loaded.get_character_by_name("小不点").id == "c1"
        assert len(loaded.scenes) == 2
        assert loaded.scenes[0].assets.image_status == AssetStatus.COMPLETE
        assert loaded.scenes[0].assets.image_path == "img/s1.png"
        assert loaded.scenes[1].assets.image_status == AssetStatus.PENDING
        assert loaded.total_cost() == 0.15

    def test_manifest_json_is_readable(self, manager, tmp_project):
        """Verify the saved JSON is human-readable."""
        manager.create(project_name="test")
        raw = (tmp_project / "manifest.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        assert data["project_name"] == "test"
        # Should be pretty-printed
        assert "\n" in raw
