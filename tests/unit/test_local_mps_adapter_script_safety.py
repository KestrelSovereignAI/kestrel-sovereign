"""Security contracts for the Local MPS generation subprocess script."""

import ast
import json
import os
from pathlib import Path

import pytest

from kestrel_sovereign.features.training.adapters import local_mps_adapter
from kestrel_sovereign.features.training.adapters import (
    _local_mps_generation_lifecycle as generation_lifecycle,
)
from kestrel_sovereign.features.training.types import GenerationConfig, GenerationState


def test_generation_script_keeps_dynamic_values_out_of_executable_source():
    prompt = (
        'PROMPT_MARKER "quoted" \\backslash\nraise AssertionError("prompt injection")'
    )
    model_path = Path(
        'MODEL_MARKER "quoted" \\backslash\n); raise AssertionError("model injection"); #'
    )
    lora_path = (
        'LORA_MARKER "quoted" \\backslash\n); raise AssertionError("lora injection"); #'
    )
    script, serialized_payload = local_mps_adapter._build_generation_script(
        model_path=model_path,
        lora_path=lora_path,
        lora_fd=None,
        output_fd=41,
        prompt=prompt,
        num_inference_steps=23,
        guidance_scale=6.25,
        width=768,
        height=512,
    )

    compile(script, "<local-mps-generation>", "exec")
    tree = ast.parse(script)
    assert not any(isinstance(node, ast.JoinedStr) for node in ast.walk(tree))
    assert "json.loads(sys.argv[1])" in script

    dynamic_values = (str(model_path), lora_path, prompt)
    assert all(value not in script for value in dynamic_values)
    assert json.loads(serialized_payload) == {
        "model_path": str(model_path),
        "lora_path": lora_path,
        "lora_fd": None,
        "output_fd": 41,
        "prompt": prompt,
        "num_inference_steps": 23,
        "guidance_scale": 6.25,
        "width": 768,
        "height": 512,
    }


@pytest.mark.asyncio
async def test_generate_image_passes_script_values_as_json_argv(tmp_path, monkeypatch):
    model_path = tmp_path / 'MODEL_ARG "quoted" \\backslash\nmodel injection; #'
    working_dir = tmp_path / 'OUTPUT_ARG "quoted" \\backslash\noutput injection; #'
    diffusers_path = tmp_path / "diffusers"
    diffusers_python = diffusers_path / ".venv/bin/python3"
    diffusers_python.parent.mkdir(parents=True)
    diffusers_python.write_text("#!/usr/bin/env python3", encoding="utf-8")
    lora_path = (
        tmp_path / 'LORA_ARG "quoted" \\backslash\nlora injection; #.safetensors'
    )
    lora_path.write_bytes(b"lora")
    prompt = 'PROMPT_ARG "quoted" \\backslash\nraise AssertionError("source injection")'

    adapter = local_mps_adapter.LocalMPSTrainingAdapter(
        model_path=str(model_path),
        working_dir=str(working_dir),
        diffusers_path=str(diffusers_path),
    )
    captured_args: tuple[str, ...] | None = None
    captured_kwargs = None
    captured_workspace = None

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            assert captured_args is not None
            payload = json.loads(captured_args[3])
            os.ftruncate(payload["output_fd"], 0)
            os.pwrite(payload["output_fd"], b"png-bytes", 0)
            return b"OK", b""

    async def fake_create_subprocess_exec(*args, **kwargs):
        nonlocal captured_args, captured_kwargs
        captured_args = args
        captured_kwargs = kwargs
        return FakeProcess()

    real_create_workspace = generation_lifecycle.create_generation_workspace

    async def track_workspace(root):
        nonlocal captured_workspace
        lease = await real_create_workspace(root)
        captured_workspace = lease.path
        return lease

    monkeypatch.setattr(
        local_mps_adapter.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        generation_lifecycle,
        "create_generation_workspace",
        track_workspace,
    )

    result = await adapter.generate_image(
        config=GenerationConfig(
            prompt=prompt,
            lora_path=str(lora_path),
            width=640,
            height=384,
            num_inference_steps=17,
            guidance_scale=5.5,
        )
    )

    assert result.state is GenerationState.COMPLETED
    assert result.images == ["data:image/png;base64,cG5nLWJ5dGVz"]

    assert captured_args is not None
    assert captured_args[:2] == (str(diffusers_python), "-c")
    assert captured_kwargs is not None
    if os.name == "posix":
        assert captured_kwargs["start_new_session"] is True
        assert len(captured_kwargs["pass_fds"]) == 1
    script, serialized_payload = captured_args[2:]
    compile(script, "<captured-local-mps-generation>", "exec")
    tree = ast.parse(script)
    assert not any(isinstance(node, ast.JoinedStr) for node in ast.walk(tree))
    assert all(
        value not in script
        for value in (str(model_path), str(lora_path), str(working_dir), prompt)
    )
    payload = json.loads(serialized_payload)
    output_fd = payload.pop("output_fd")
    assert payload == {
        "model_path": str(model_path),
        "lora_path": str(lora_path),
        "lora_fd": None,
        "prompt": prompt,
        "num_inference_steps": 17,
        "guidance_scale": 5.5,
        "width": 640,
        "height": 384,
    }
    assert output_fd in captured_kwargs["pass_fds"]
    assert captured_workspace is not None
    assert captured_workspace.parent == working_dir.resolve()
    assert captured_workspace.name.startswith(
        generation_lifecycle.GENERATION_WORKSPACE_PREFIX
    )
    assert not captured_workspace.exists()
