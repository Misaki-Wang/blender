"""Offline orchestration checks; no API requests, Blender runs or Slurm jobs."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from PIL import Image


class FakeClient:
    calls = []
    root = None
    omit_description = False

    def __init__(self, *args, **kwargs):
        self.http = SimpleNamespace(get=lambda *a, **kw: SimpleNamespace(raise_for_status=lambda: None))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def capabilities(self):
        return {"reserve_conversation": True, "workspace_write_roots": [str(self.root)]}

    def create_conversation(self):
        return {"conversation_id": str(uuid4())}

    @staticmethod
    def image(path):
        return {"path": str(path)}

    def submit(self, prompt, **options):
        self.calls.append((prompt, options))
        return {"id": str(len(self.calls))}

    def wait(self, job_id, **kwargs):
        prompt, options = self.calls[int(job_id) - 1]
        output = Path(options["workspace_dir"])
        if not self.omit_description and ("Save your concise scene observations" in prompt or "Describe the scene in the attached" in prompt):
            (output / "scene_description").write_text("A red apple with a stem on a table.\n")
        if "Required output file:" in prompt:
            (output / "blender_code.py").write_text("import bpy\ndef build_scene():\n    return bpy.context.scene\n")
        return {"id": job_id, "status": "completed", "response": {
            "conversation_id": options.get("conversation_id") or str(uuid4()),
            "answer": "Saved files.", "usage": {}}}


def main():
    # Supply only the queued client boundary; exercise the real orchestration.
    client_module = ModuleType("api_server.client")
    client_module.CodexClient = FakeClient
    with patch.dict(sys.modules, {"api_server": ModuleType("api_server"), "api_server.client": client_module}):
        import blender_gen
        import asset_gen

    with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        root = Path(temporary)
        FakeClient.root = root
        image = root / "reference.png"
        Image.new("RGB", (16, 16), "red").save(image)
        for mode, supplied, count in (("combined", None, 1), ("separate", None, 2), ("combined", "An apple.", 1)):
            FakeClient.calls = []
            result = blender_gen.generate_one(None, image, output_dir=root, scene_description=supplied,
                                               description_mode=mode)
            assert len(FakeClient.calls) == count, (mode, len(FakeClient.calls))
            assert Path(result["code_path"]).is_file() and Path(result["scene_description_path"]).is_file()
            assert result["syntax_valid"]
            if mode == "combined" and supplied is None:
                assert result["description_conversation_id"] == result["code_conversation_id"]
            if supplied:
                assert Path(result["scene_description_path"]).read_text().strip() == supplied

        FakeClient.calls = []
        FakeClient.omit_description = True
        try:
            blender_gen.generate_one(None, image, output_dir=root, description_mode="combined")
        except ValueError as error:
            assert "场景描述" in str(error)
        else:
            raise AssertionError("Missing combined description must fail")
        FakeClient.omit_description = False

        source = root / "source"
        source.mkdir()
        (source / "scene.blend").write_bytes(b"source")
        output = root / "asset" / "apple" / "0" / "0"
        output.mkdir(parents=True)
        for name in ("0_only_color.blend", "0_asset_preview_color.png"):
            (output / name).write_bytes(b"existing")
        args = SimpleNamespace(source_dir=source, output_root=root / "asset", pipeline_root=root,
                               category="apple", sample="0", generate_only=False, extract_script=None)
        with patch.object(asset_gen, "parse_args", return_value=args), patch.object(asset_gen, "generate_script") as generate_script:
            assert asset_gen.main() == 0
            generate_script.assert_not_called()
            assert json.loads((output / "status.json").read_text())["script_generation"] == "skipped_existing_outputs"
            args.generate_only = True
            runner = root / "runner.py"
            runner.write_text("pass\n")
            generate_script.return_value = runner
            assert asset_gen.main() == 0
            generate_script.assert_called_once()
    print("PASS: combined=1 request; separate=2; supplied=1; missing description rejected; asset reuse=0; generate-only retained")


if __name__ == "__main__":
    main()
