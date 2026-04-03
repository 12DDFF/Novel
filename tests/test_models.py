"""Tests for data models and enums."""
import json

from src.models import (
    AssetStatus,
    Character,
    Manifest,
    Mood,
    Scene,
    SceneAssets,
    Settings,
    Source,
    TransitionType,
)


class TestCharacter:
    def test_create_with_defaults(self):
        char = Character(name="王林")
        assert char.name == "王林"
        assert char.aliases == []
        assert char.id.startswith("char_")
        assert char.reference_image_path is None

    def test_create_with_full_data(self):
        char = Character(
            name="王林",
            aliases=["小林", "林哥"],
            description="tall cultivator with white robes",
            role="protagonist",
            image_prompt_prefix="a tall Chinese man in white robes",
        )
        assert char.aliases == ["小林", "林哥"]
        assert char.role == "protagonist"

    def test_unique_ids(self):
        c1 = Character(name="A")
        c2 = Character(name="B")
        assert c1.id != c2.id


class TestScene:
    def test_create_with_defaults(self):
        scene = Scene(sequence=1, narration_text="他走进了黑暗的巷子。")
        assert scene.sequence == 1
        assert scene.mood == Mood.DRAMATIC
        assert scene.transition == TransitionType.CROSSFADE
        assert scene.assets.image_status == AssetStatus.PENDING
        assert scene.assets.audio_status == AssetStatus.PENDING

    def test_scene_with_full_data(self):
        scene = Scene(
            sequence=5,
            narration_text="剑光一闪，敌人倒下了。",
            visual_description="sword flash, enemy falls",
            characters_present=["char_001", "char_002"],
            mood=Mood.ACTION,
            setting="battlefield at dawn",
            image_prompt="cinematic anime, battlefield, dawn",
            transition=TransitionType.CUT,
            duration_estimate_seconds=6.5,
        )
        assert scene.mood == Mood.ACTION
        assert len(scene.characters_present) == 2
        assert scene.duration_estimate_seconds == 6.5


class TestSceneAssets:
    def test_defaults_all_pending(self):
        assets = SceneAssets()
        assert assets.image_status == AssetStatus.PENDING
        assert assets.audio_status == AssetStatus.PENDING
        assert assets.subtitle_status == AssetStatus.PENDING
        assert assets.image_path is None

    def test_update_status(self):
        assets = SceneAssets()
        assets.image_status = AssetStatus.COMPLETE
        assets.image_path = "data/images/scenes/scene_001.png"
        assert assets.image_status == AssetStatus.COMPLETE
        assert assets.image_path == "data/images/scenes/scene_001.png"


class TestManifest:
    def _make_manifest(self) -> Manifest:
        chars = [
            Character(id="char_001", name="王林", aliases=["小林"]),
            Character(id="char_002", name="李慕婉"),
        ]
        scenes = [
            Scene(id="scene_001", sequence=1, characters_present=["char_001"]),
            Scene(id="scene_002", sequence=2, characters_present=["char_001", "char_002"]),
            Scene(id="scene_003", sequence=3, characters_present=["char_002"]),
        ]
        return Manifest(
            project_name="仙逆",
            source=Source(platform="fanqie", novel_title="仙逆"),
            characters=chars,
            scenes=scenes,
        )

    def test_create_default(self):
        m = Manifest()
        assert m.project_id
        assert m.characters == []
        assert m.scenes == []
        assert m.total_cost() == 0.0

    def test_get_character_by_id(self):
        m = self._make_manifest()
        char = m.get_character("char_001")
        assert char is not None
        assert char.name == "王林"
        assert m.get_character("nonexistent") is None

    def test_get_character_by_name(self):
        m = self._make_manifest()
        assert m.get_character_by_name("王林") is not None
        assert m.get_character_by_name("小林") is not None  # alias
        assert m.get_character_by_name("张三") is None

    def test_get_scene(self):
        m = self._make_manifest()
        scene = m.get_scene("scene_002")
        assert scene is not None
        assert scene.sequence == 2

    def test_get_scenes_by_status(self):
        m = self._make_manifest()
        pending = m.get_scenes_by_status(AssetStatus.PENDING, "image")
        assert len(pending) == 3

        m.scenes[0].assets.image_status = AssetStatus.COMPLETE
        pending = m.get_scenes_by_status(AssetStatus.PENDING, "image")
        assert len(pending) == 2
        complete = m.get_scenes_by_status(AssetStatus.COMPLETE, "image")
        assert len(complete) == 1

    def test_cost_tracking(self):
        m = self._make_manifest()
        m.add_cost("gpu", 0.15, "RunPod RTX4090 30min")
        m.add_cost("llm", 0.01, "DeepSeek segmentation")
        assert m.total_cost() == 0.16
        assert len(m.costs) == 2

    def test_progress_summary(self):
        m = self._make_manifest()
        m.scenes[0].assets.image_status = AssetStatus.COMPLETE
        m.scenes[1].assets.image_status = AssetStatus.GENERATING
        summary = m.progress_summary()
        assert summary["image"]["complete"] == 1
        assert summary["image"]["generating"] == 1
        assert summary["image"]["pending"] == 1
        assert summary["audio"]["pending"] == 3

    def test_serialization_roundtrip(self):
        m = self._make_manifest()
        m.add_cost("gpu", 0.15)
        m.scenes[0].assets.image_status = AssetStatus.COMPLETE

        # Serialize to JSON
        json_str = m.model_dump_json(indent=2)
        data = json.loads(json_str)

        # Deserialize back
        m2 = Manifest.model_validate(data)
        assert m2.project_name == "仙逆"
        assert len(m2.characters) == 2
        assert len(m2.scenes) == 3
        assert m2.scenes[0].assets.image_status == AssetStatus.COMPLETE
        assert m2.total_cost() == 0.15
        assert m2.get_character_by_name("小林") is not None
