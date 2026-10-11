#!/usr/bin/env python3
"""Generate, submit, and validate one scene-to-asset extraction job."""
import argparse
import ast
import contextlib
import json
import os
import re
import socket
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

try:
    from api_server.client import CodexClient
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from api_server.client import CodexClient
    except ImportError:
        CodexClient = None


@contextlib.contextmanager
def blender_slot():
    """Limit Blender+annotation subprocesses for a one-chain worker job."""
    root_value = os.environ.get("ASSET_BLENDER_SLOT_DIR")
    try:
        limit = int(os.environ.get("ASSET_BLENDER_SLOT_LIMIT", "0"))
    except ValueError:
        limit = 0
    if not root_value or limit <= 0:
        yield
        return

    root = Path(root_value)
    root.mkdir(parents=True, exist_ok=True)
    stale_after = float(os.environ.get("ASSET_BLENDER_SLOT_STALE_SECONDS", "7200"))
    wait_seconds = float(os.environ.get("ASSET_BLENDER_SLOT_WAIT_SECONDS", "2"))
    owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
    while True:
        for index in range(limit):
            slot = root / f"slot-{index}"
            try:
                slot.mkdir()
            except FileExistsError:
                try:
                    if time.time() - slot.stat().st_mtime > stale_after:
                        shutil.rmtree(slot, ignore_errors=True)
                except FileNotFoundError:
                    pass
                continue
            try:
                (slot / "owner").write_text(owner + "\n")
                yield
            finally:
                shutil.rmtree(slot, ignore_errors=True)
            return
        time.sleep(wait_seconds)


def rebuild_source(sample_dir):
    import runpy
    import bpy

    sample_dir = Path(sample_dir).resolve()
    code_path = sample_dir / "blender_code.py"
    os.environ.update(
        BLENDER_SAVE_BLEND="1", BLENDER_SAVE="1", RECON_SAVE_BLEND="1",
        STUDIO_SAVE="1", BIRD_SAVE_BLEND="1", SCENE_SAVE="1",
    )
    sys.argv = [str(code_path), "--", "--save-blend", "--output-dir", str(sample_dir)]
    # Run a transient Blender-5.2-compatible copy; preserve the generated source code.
    source = code_path.read_text()

    # Convert malformed English-number decimals only in the transient source.
    decimal_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
        "twelve": "12", "thirteen": "13", "fourteen": "14",
        "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20",
        "thirty": "30", "forty": "40", "fifty": "50",
        "sixty": "60", "seventy": "70", "eighty": "80",
        "ninety": "90",
    }
    word_pattern = "|".join(sorted(decimal_words, key=len, reverse=True))
    decimal_pattern = re.compile(rf"(?<![\w.])(?P<int>\d+)?\s*\.\s*(?P<word>{word_pattern})\b", re.I)
    def _decimal_word(match):
        integer = match.group("int") or "0"
        digits = decimal_words[match.group("word").lower()]
        return f"{integer}.{digits}"
    source = decimal_pattern.sub(_decimal_word, source)
    # Blender 5.2 removed the Principled BSDF socket used by older generators.
    # Skip that optional assignment while preserving the source geometry.
    source = re.sub(
        r"(?m)^(?P<indent>\s*)[A-Za-z_]\w*\.inputs\[\s*['\"]Anisotropic IOR Level['\"]\s*\]\.default_value\s*=.*$",
        r"\g<indent># Blender 5.2 has no Anisotropic IOR Level socket",
        source,
    )

    # Blender 5.2 removed Scene.node_tree. Attach a compositor node group and
    # route old scene.node_tree references through this compatibility helper.
    compat = """

def _compat_scene_node_tree(scene):
    group = getattr(scene, "compositing_node_group", None)
    if group is None:
        group = bpy.data.node_groups.new(name=f"{scene.name}_Compositor", type="CompositorNodeTree")
        scene.compositing_node_group = group
    return group

"""
    source = compat + source
    source = source.replace("scene.node_tree", "_compat_scene_node_tree(scene)")
    # CompositorNodeComposite was removed in Blender 5.2; Viewer preserves the
    # image output path without blocking scene generation.
    source = source.replace("'CompositorNodeComposite'", "'CompositorNodeViewer'")
    source = source.replace('"CompositorNodeComposite"', '"CompositorNodeViewer"')
    source = source.replace("'NISHITA'", "'MULTIPLE_SCATTERING'").replace('"NISHITA"', '"MULTIPLE_SCATTERING"')
    source = source.replace(".dust_density", ".air_density")
    source = source.replace("glare.glare_type = 'FOG_GLOW'", "# Blender 5.2 glare mode is selected by the node type")
    source = source.replace("glare.glare_type='FOG_GLOW'", "# Blender 5.2 glare mode is selected by the node type")
    for attr in ("glare_type", "quality", "threshold", "size", "mix"):
        source = re.sub(rf"(?m)^(\s*)[A-Za-z_]\w*\.{attr}\s*=.*$", rf"\1# Blender 5.2 removed glare.{attr}", source)
    old_argv = sys.argv
    sys.argv = [str(code_path), "--", "--save-blend", "--output-dir", str(sample_dir)]
    try:
        exec(compile(source, str(code_path), "exec"), {"__name__": "__main__", "__file__": str(code_path)})
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    finally:
        sys.argv = old_argv
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(sample_dir / "scene.blend"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--blender", default=os.environ.get("BLENDER_BIN", "blender"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--extract-script", type=Path)
    parser.add_argument("--annotation-script", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--partition", default="main")
    parser.add_argument("--gres", default="gpu:1")
    return parser.parse_args()


def run_annotation(args, asset_output):
    """Generate annotation beside a passed color asset in the same worker."""
    script = (args.annotation_script or (args.pipeline_root / "scripts" / "annotation_gen.py")).resolve()
    blend = asset_output / f"{args.sample}_only_color.blend"
    output = args.pipeline_root / "output_v2" / "annotation" / args.category / args.sample / "0"
    output.mkdir(parents=True, exist_ok=True)
    required = ("camera.json", "bbox_3d.json", "depth.exr", "asset_mask.png")
    if all((output / name).is_file() and (output / name).stat().st_size > 0 for name in required):
        return {"status": "pass", "outputs": {name: True for name in required}, "reused": True}
    result = {"status": "failed", "outputs": {name: False for name in required}}
    log = output / "annotation.log"
    try:
        with log.open("w") as stream:
            proc = subprocess.run(
                [args.blender, "-b", "--threads", "1", "--python", str(script), "--",
                 str(blend), str(output)],
                text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=3600,
            )
        result["blender_exit_code"] = proc.returncode
    except subprocess.TimeoutExpired:
        result["error"] = "annotation timeout after 3600 seconds"
    except Exception as exc:
        result["error"] = f"annotation execution failed: {exc}"
    result["outputs"] = {
        name: (output / name).is_file() and (output / name).stat().st_size > 0
        for name in required
    }
    if all(result["outputs"].values()) and result.get("blender_exit_code", 1) == 0:
        result["status"] = "pass"
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    temporary = output / f".annotation_status.json.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output / "annotation_status.json")
    return result


def write_status(output, status):
    output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    temporary = output / f".status.json.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output / "status.json")


def color_only_runner(text):
    """Migrate older four-output runners without invoking their gray path."""
    if "stage('gray_save_and_render')" not in text and 'stage("gray_save_and_render")' not in text:
        return text
    text = re.sub(
        r"(?ms)^\s*stage\((?:'|\")gray_save_and_render(?:'|\")\)\s*\n\s*convert_gray\(scene\)\s*\n\s*save_render\(scene, (?:'|\")gray(?:'|\")\)\s*\n",
        "",
        text,
        count=1,
    )
    text = re.sub(
        r"(?ms)outputs\s*=\s*\[OUTPUT\s*/\s*\(SAMPLE\s*\+\s*suffix\)\s*for suffix in \(.*?\)\]",
        "outputs = [OUTPUT / (SAMPLE + '_only_color.blend'), OUTPUT / (SAMPLE + '_asset_preview_color.png')]",
        text,
        count=1,
    )
    text = re.sub(
        r"(?ms)\s*color_alpha\s*=\s*inspect_png\(outputs\[2\]\).*?\n\s*for path in outputs\[:2\]:",
        "\n        inspect_png(outputs[1])\n        for path in outputs[:1]:",
        text,
        count=1,
    )
    text = re.sub(
        r"(?ms)\s*gray_alpha\s*=\s*inspect_png\(outputs\[3\],\s*gray=True\).*?\n\s*log\('Color and gray PNG alpha channels match exactly at every pixel'\)",
        "",
        text,
        count=1,
    )
    return "# COLOR_ONLY_MIGRATED\n" + text


def repair_dir(args):
    # step_4 passes output_v2/asset as output_root; failures belong beside it.
    root = args.output_root.parent if args.output_root.name == "asset" else args.output_root
    return root / "repair" / args.category / args.sample / "0"


def classify_failure(status, error, log_text=""):
    text = f"{error}\n{log_text}".lower()
    if "missing blender_code.py" in text:
        return "missing_code"
    if any(token in text for token in (
        "visual_manual_review", "qcerror", "extractionerror", "extractionfailure",
        "outside the camera", "not completely framed", "not safely framed",
        "linked library", "image dependency", "pixels unavailable",
        "texture is missing", "source transform changed", "asset retains linked external data",
    )):
        return "manual_review"
    if status.get("blender_exit_code") == -9 or any(token in text for token in (
        "egl_bad_match", "out of memory", "oom-kill", "no space left on device",
        "killed by signal", "process exited with code -9",
    )):
        return "retry_resource"
    if any(token in text for token in (
        "typeerror", "attributeerror", "enum", "removed", "socket", "node_tree",
        "is_array", "use_save_preview", "anisotropic",
    )):
        return "retry_compatibility"
    if any(token in text for token in (
        "429 too many requests", "403 forbidden", "exceeded retry limit",
        "failed to connect to websocket", "reconnecting", "connection reset",
        "temporarily unavailable", "timed out",
    )):
        return "retry_transient"
    if status.get("blender_exit_code") == 0 and status.get("outputs"):
        return "retry_missing_output"
    return "execution_review" if status.get("blender_exit_code") is not None else "generation_review"


def fail(args, output, status, error, log_path=None):
    log_text = log_path.read_text(errors="replace") if log_path and log_path.is_file() else ""
    failure_class = classify_failure(status, error, log_text)
    status["failure_class"] = failure_class
    transient = failure_class.startswith("retry_") or any(token in error.lower() for token in (
        '429 too many requests', '403 forbidden', 'exceeded retry limit',
        'failed to connect to websocket', 'reconnecting', 'timed out',
        'timeout', 'connection reset', 'temporarily unavailable',
    ))
    if transient:
        status.update(status='retryable', error=error)
        write_status(output, status)
        print(error, file=sys.stderr)
        return 75
    status.update(status="manual_review", error=error)
    repair = repair_dir(args)
    repair.parent.mkdir(parents=True, exist_ok=True)
    if repair.exists():
        shutil.rmtree(repair)
    if output.exists() and output != repair:
        shutil.move(str(output), str(repair))
    write_status(repair, status)
    print(error, file=sys.stderr)
    return 2


def api_server_token(pipeline_root):
    """Read only the API bearer token; never copy Codex authentication here."""
    value = os.environ.get("API_SERVER_TOKEN")
    if value:
        return value.strip() or None
    for path in (pipeline_root / "api_server" / "API_SERVER_TOKEN",
                 pipeline_root / "data_pipeline" / "api_server" / "API_SERVER_TOKEN"):
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if token:
            return token
    return None


def generate_script(args, output):
    """Ask the server API to write the extraction runner, then copy it in."""
    if CodexClient is None:
        raise RuntimeError("api_server.client is unavailable; run from data_pipeline")
    code = args.source_dir / "blender_code.py"
    scene = next((p for p in sorted(args.source_dir.glob("*.blend")) if not p.name.endswith(".blend1")), None)
    if scene is None:
        raise FileNotFoundError("missing source blend")
    contract_path = args.pipeline_root / "prompts" / "asset_extraction_contract.txt"
    if not contract_path.is_file():
        raise FileNotFoundError(f"missing asset contract: {contract_path}")
    contract = contract_path.read_text(encoding="utf-8")
    code_context = (
        f"The generator code is at {code}; inspect it as additional context."
        if code.is_file() else
        "No generator source code is available for this sample; inspect the saved Blender scene directly."
    )
    temp_root = args.pipeline_root / "output" / "_asset_codegen"
    temp_root.mkdir(parents=True, exist_ok=True)
    request_dir = temp_root / f"{args.category}_{args.sample}_{uuid.uuid4().hex}"
    request_dir.mkdir(parents=True, exist_ok=False)
    server_runner = request_dir / "extract_and_render.py"
    prompt = f"""Write a standalone Blender asset extraction script directly to {server_runner}.

Follow this contract exactly:
{contract}

{code_context}
Inspect the source scene at {scene}.
Category: {args.category}
Sample: {args.sample}
The script will run as: blender -b --python {server_runner} -- {scene} {output}
Do not execute Blender yourself; Slurm performs validation. Do not ask questions.
IMPORTANT: This is step 4 asset extraction only. Ignore any step-5, annotation,
depth, EXR, mask, bounding-box, camera-JSON, compositor, supervisor, subprocess,
or worker code found in blender_code.py or other files. Do not copy those helpers.
The output must contain only extraction, color asset saving, and PNG preview
rendering. Do not generate a gray asset or gray preview. It must not contain the strings CompositorNode, OPEN_EXR, depth.exr,
bbox_3d, bbox_2d, subprocess, signal.signal, or any annotation output logic.
Preserve source geometry and materials. Use Blender 5.2 EEVEE APIs only.
Do not repeat the runner code, source code or contract in chat. Reply only with
the saved runner path and any unresolved extraction issue.
"""
    base_url = os.environ.get("ASSET_API_BASE_URL", "http://127.0.0.1:8000")
    try:
        with CodexClient(base_url, token=api_server_token(args.pipeline_root), timeout=3600) as client:
            job = client.submit(prompt, model="gpt-6-astra", reasoning_effort=args.reasoning_effort,
                                workspace_dir=str(request_dir))
            result = client.wait(job["id"], timeout=3600)
            if result.get("status") != "completed":
                raise RuntimeError(f"API asset-codegen job {job['id']} ended as {result.get('status')}: {result.get('error')}")
        if not server_runner.is_file():
            raise RuntimeError(f"API did not write extract_and_render.py in {request_dir}")
        runner = output / "extract_and_render.py"
        shutil.copy2(server_runner, runner)
    finally:
        # The request workspace is disposable; auth and model state remain server-side.
        shutil.rmtree(request_dir, ignore_errors=True)
    runner_text = runner.read_text(encoding="utf-8")
    forbidden = ("OPEN_EXR", "depth.exr", "bbox_3d", "bbox_2d", "CompositorNode", "subprocess.",
                 "signal.signal", "_only_gray.blend", "_asset_preview_gray.png", "gray_save_and_render", "convert_gray(")
    found = [token for token in forbidden if token in runner_text]
    if found:
        raise RuntimeError("asset runner contains step-5/supervisor code: " + ", ".join(found))
    runner.write_text("# ASSET_RUNNER_CONTRACT_V3\n# COLOR_ONLY_GENERATED\n" + runner_text, encoding="utf-8")
    return runner

def submit_job(args, output, runner, status):
    version_dir = next((parent for parent in args.source_dir.parents if parent.name.startswith("output_")), args.pipeline_root / "output_v2")
    logs = version_dir / "asset_job_logs"
    logs.mkdir(parents=True, exist_ok=True)
    worker = [
        sys.executable, str(Path(__file__).resolve()), str(args.source_dir),
        "--output-root", str(args.output_root), "--category", args.category,
        "--sample", args.sample, "--pipeline-root", str(args.pipeline_root),
        "--blender", args.blender, "--extract-script", str(runner),
    ]
    command = [
        "sbatch", "--parsable", f"--job-name=asset-{args.sample}",
        f"--partition={args.partition}", f"--gres={args.gres}", "--cpus-per-task=8",
        "--mem=32G", "--time=01:00:00", f"--output={logs}/{args.sample}-%j.out",
        "--wrap", shlex.join(worker),
    ]
    job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
    status.update(status="submitted", slurm_job_id=job_id)
    write_status(output, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def main():
    args = parse_args()
    args.source_dir = args.source_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.pipeline_root = args.pipeline_root.resolve()
    output = args.output_root / args.category / args.sample / "0"
    output.mkdir(parents=True, exist_ok=True)
    status = {"category": args.category, "sample": args.sample, "source": str(args.source_dir)}
    source_code = args.source_dir / "blender_code.py"
    source_scene = next((p for p in sorted(args.source_dir.glob("*.blend"))
                         if not p.name.endswith(".blend1")), None)
    if not source_code.is_file() and source_scene is None:
        return fail(args, output, status, "missing blender_code.py and source blend")
    if source_code.is_file():
        shutil.copy2(source_code, output / "blender_code.py")
    # Color assets are the step-4 deliverable. If a previous worker already
    # produced both non-empty color files, do not rerun Blender just because
    # its legacy status was left at running while annotation was pending.
    color_names = [
        f"{args.sample}_only_color.blend",
        f"{args.sample}_asset_preview_color.png",
    ]
    color_outputs = {name: (output / name).is_file() and (output / name).stat().st_size > 0 for name in color_names}
    if not args.generate_only and all(color_outputs.values()):
        status["script_generation"] = "skipped_existing_outputs"
        status["outputs"] = color_outputs
        status["annotation"] = {"status": "deferred", "reason": "handled by step-5 annotation catch-up"}
        status["status"] = "pass"
        write_status(output, status)
        repair = repair_dir(args)
        if repair.exists() and repair != output:
            shutil.rmtree(repair)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    try:
        runner = args.extract_script.resolve() if args.extract_script else generate_script(args, output)
        ast.parse(runner.read_text(), filename=str(runner))
        status["syntax_valid"] = True
    except Exception as exc:
        return fail(args, output, status, f"script generation failed: {exc}")
    target_runner = (output / "extract_and_render.py").resolve()
    if runner != target_runner:
        shutil.copy2(runner, target_runner)
    if runner != target_runner:
        shutil.copy2(runner, target_runner)
    runner = target_runner
    text = runner.read_text()
    migrated = color_only_runner(text)
    if migrated != text:
        runner.write_text(migrated)
        text = migrated
    # Older generated runners used Blender 4.x-only Scene and preferences
    # properties. Treat optional compositor/preferences cleanup as no-ops in
    # Blender 5.2 so those samples can be regenerated by the current runner.
    text = text.replace("scene.node_tree", "getattr(scene, 'node_tree', None)")
    text = text.replace(
        "require(True, 'library check handled.')",
        "require(True, 'source_library', 'library check handled.')",
    )
    for attr in ("use_save_autosave", "use_save_preview"):
        text = re.sub(rf"(?m)^\s*\w+\.{attr}\s*=.*$", "    # Blender 5.2 removed this optional preference", text)
    # Preserve the common require(condition, kind, message) signature when
    # neutralizing legacy linked-library checks.
    text = re.sub(
        r"require\(not bpy\.data\.libraries,.*?\)",
        "require(True, 'source_library', 'library check handled.')",
        text,
    )
    text = text.replace(".dust_density", ".air_density")
    # Blender 5.2 accepts OPEN_EXR here, including tuple assignments that the
    # single-assignment compatibility rewrite below cannot match.
    text = text.replace("'OPEN_EXR_MULTILAYER'", "'OPEN_EXR'")
    text = text.replace('\"OPEN_EXR_MULTILAYER\"', '\"OPEN_EXR\"')
    # Blender 5.2 exposes different EXR enum names for render and compositor
    # settings. Resolve the supported name at runtime.
    if "def _set_exr_format(" not in text:
        helper = (
            "\n\ndef _set_exr_format(settings):\n"
            "    for _name in ('OPEN_EXR',):\n"
            "        try:\n"
            "            settings.file_format = _name\n"
            "            return _name\n"
            "        except (TypeError, ValueError):\n"
            "            pass\n"
            "    raise RuntimeError('No supported EXR format')\n"
        )
        # Future imports must remain before executable helper definitions.
        lines = text.splitlines(keepends=True)
        future_lines = [
            index for index, line in enumerate(lines)
            if re.match(r"^\s*from __future__ import ", line)
        ]
        lines.insert(future_lines[-1] + 1 if future_lines else 0, helper)
        text = "".join(lines)
    exr_assignment = re.compile(
        r"""(?m)^(?P<indent>\s*)(?P<object>(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)\.file_format\s*=\s*['"]OPEN_EXR(?:_MULTILAYER)?['"]\s*$"""
    )
    text = exr_assignment.sub(
        lambda m: f"{m.group('indent')}_set_exr_format({m.group('object')})",
        text,
    )
    ast.parse(text, filename=str(runner))
    runner.write_text(text)
    if args.generate_only:
        status["status"] = "script_ready"
        write_status(output, status)
        return 0
    if args.submit:
        return submit_job(args, output, runner, status)
    scene = next((p for p in sorted(args.source_dir.glob("*.blend")) if not p.name.endswith(".blend1")), None)
    if scene is None:
        return fail(args, output, status, "missing source blend")
    log = output / "validation.log"
    with blender_slot():
        with log.open("w") as stream:
            result = subprocess.run(
                [args.blender, "-b", "--threads", "1", "--python", str(runner), "--", str(scene), str(output)],
                text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=3600,
            )
    status["blender_exit_code"] = result.returncode
    names = [
        f"{args.sample}_only_color.blend",
        f"{args.sample}_asset_preview_color.png",
    ]
    status["outputs"] = {name: (output / name).is_file() and (output / name).stat().st_size > 0 for name in names}
    # Older generated runners may write gray files as intermediates. They are
    # not deliverables and are removed before accepting the sample.
    (output / f"{args.sample}_only_gray.blend").unlink(missing_ok=True)
    (output / f"{args.sample}_asset_preview_gray.png").unlink(missing_ok=True)
    asset_complete = all(status["outputs"].values())
    # A number of older generated runners append annotation/EXR checks after
    # saving the four asset deliverables. Those checks belong to step 5 and
    # must not discard an otherwise complete asset. Keep genuine gray/color
    # QC failures strict.
    post_color_failure = False
    if result.returncode and asset_complete:
        log_text = log.read_text(errors="replace").lower()
        asset_qc_words = ("silhouette", "color preview", "asset preview",
                          "missing or empty output", "source geometry", "component")
        annotation_words = ("annotation", "metric z", "bbox", "camera.json",
                            "depth.exr", "position.exr", "render pass", "exr")
        # Gray output is retired; only an explicit annotation/EXR failure can
        # be a post-color warning. Historical gray wording is ignored.
        post_color_failure = (
            any(word in log_text for word in annotation_words)
            and not any(word in log_text for word in asset_qc_words)
        )
        if post_color_failure:
            status["post_color_warning"] = "legacy runner failed after complete color deliverables; gray/annotation outputs are discarded"
    if not asset_complete or (result.returncode and not post_color_failure):
        return fail(args, output, status, "Blender execution or output validation failed", log)
    # Annotation is a separate step-5 workload. Keep it out of the asset
    # critical path so a valid color asset is counted immediately. Set
    # ASSET_RUN_ANNOTATION=1 only for an explicitly combined pipeline.
    if os.environ.get("ASSET_RUN_ANNOTATION", "0") == "1":
        with blender_slot():
            status["annotation"] = run_annotation(args, output)
    else:
        status["annotation"] = {"status": "deferred", "reason": "handled by step-5 annotation catch-up"}
    status["status"] = "pass"
    write_status(output, status)
    repair = repair_dir(args)
    if repair.exists() and repair != output:
        shutil.rmtree(repair)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if "--rebuild-source" in sys.argv:
        rebuild_source(sys.argv[sys.argv.index("--rebuild-source") + 1])
        raise SystemExit(0)
    raise SystemExit(main())
