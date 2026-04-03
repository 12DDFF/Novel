from __future__ import annotations

import json
from pathlib import Path

from filelock import FileLock

from src.models.manifest import Manifest


class ManifestManager:
    """Reads, writes, and updates project manifests with file locking."""

    def __init__(self, project_dir: str | Path):
        self.project_dir = Path(project_dir)
        self.manifest_path = self.project_dir / "manifest.json"
        self.lock_path = self.project_dir / "manifest.json.lock"

    def exists(self) -> bool:
        """Check if a manifest exists in this project directory."""
        return self.manifest_path.exists()

    def create(self, project_name: str = "", **kwargs) -> Manifest:
        """Create a new manifest and save it to disk."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        manifest = Manifest(project_name=project_name, **kwargs)
        self._write(manifest)
        return manifest

    def load(self) -> Manifest:
        """Load manifest from disk."""
        with FileLock(self.lock_path):
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return Manifest.model_validate(data)

    def save(self, manifest: Manifest) -> None:
        """Save manifest to disk (updates the updated_at timestamp)."""
        from datetime import datetime
        manifest.updated_at = datetime.now()
        self._write(manifest)

    def _write(self, manifest: Manifest) -> None:
        """Write manifest to disk with file locking."""
        with FileLock(self.lock_path):
            json_str = manifest.model_dump_json(indent=2)
            self.manifest_path.write_text(json_str, encoding="utf-8")

    def update(self, fn) -> Manifest:
        """
        Load manifest, apply a function to it, and save it back.

        Usage:
            manager.update(lambda m: m.scenes.append(new_scene))
        """
        manifest = self.load()
        fn(manifest)
        self.save(manifest)
        return manifest

    def update_scene_asset(
        self,
        scene_id: str,
        asset_type: str,
        status: str,
        path: str | None = None,
    ) -> Manifest:
        """Convenience method to update a scene's asset status and path."""
        def _update(manifest: Manifest):
            scene = manifest.get_scene(scene_id)
            if scene is None:
                raise ValueError(f"Scene {scene_id} not found")
            setattr(scene.assets, f"{asset_type}_status", status)
            if path is not None:
                setattr(scene.assets, f"{asset_type}_path", path)

        return self.update(_update)

    def add_cost(self, category: str, amount: float, details: str = "") -> Manifest:
        """Record a cost entry."""
        return self.update(lambda m: m.add_cost(category, amount, details))

    def delete(self) -> None:
        """Delete manifest and lock file."""
        if self.manifest_path.exists():
            self.manifest_path.unlink()
        if self.lock_path.exists():
            self.lock_path.unlink()
