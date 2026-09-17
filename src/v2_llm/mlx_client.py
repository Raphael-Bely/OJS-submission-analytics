"""
mlx_client.py — MLXClient, a drop-in alternative to llm_client.LocalHFClient
for quantized models (4-bit/8-bit mlx-community checkpoints) via Apple's MLX
framework. Same public interface as LocalHFClient: __init__(model_path,
device) and generate(system_prompt, user_prompt) -> str.

Runs the actual model in a separate, long-lived subprocess under
.venv_mlx/bin/python3 (see mlx_generate.py) rather than importing mlx_lm
directly here. Reason: mlx-lm's current PyPI release is incompatible with the
transformers version already pinned in the codenet conda env (confirmed by
testing: mlx_lm import raises inside transformers' AutoTokenizer.register()).
Downgrading transformers in the shared env to satisfy mlx-lm would risk
breaking embedding.py/llm_client.py, which are already relied on and working
- an isolated venv (.venv_mlx, requirements resolved independently) avoids
that entirely. The subprocess loads the model once and is reused across all
generate() calls, exactly like LocalHFClient reuses its in-process model.
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MLX_VENV_PYTHON = PROJECT_ROOT / ".venv_mlx" / "bin" / "python3"
MLX_GENERATE_SCRIPT = Path(__file__).resolve().parent / "mlx_generate.py"


class MLXClient:
    def __init__(self, model_path: str | Path, device: str = "mlx"):
        # `device` accepted only for interface parity with LocalHFClient -
        # MLX always targets the Metal/GPU backend itself, nothing to select.
        if not MLX_VENV_PYTHON.exists():
            raise RuntimeError(
                f"'{MLX_VENV_PYTHON}' not found - create it first:\n"
                f"  python3 -m venv .venv_mlx && .venv_mlx/bin/pip install mlx-lm"
            )

        self._proc = subprocess.Popen(
            [str(MLX_VENV_PYTHON), str(MLX_GENERATE_SCRIPT), str(model_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,  # line-buffered
        )

        ready_line = self._proc.stdout.readline().strip()
        if ready_line != "READY":
            stderr_tail = self._proc.stderr.read()
            self._proc.wait(timeout=5)
            raise RuntimeError(
                f"MLX worker failed to load '{model_path}'.\n"
                f"First line was: {ready_line!r}\n"
                f"stderr:\n{stderr_tail}"
            )
        print(f"MLX worker ready (pid {self._proc.pid}).", file=sys.stderr)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._proc.poll() is not None:
            stderr_tail = self._proc.stderr.read()
            raise RuntimeError(
                f"MLX worker process died (exit code {self._proc.returncode}).\n"
                f"stderr:\n{stderr_tail}"
            )

        req = {"system_prompt": system_prompt, "user_prompt": user_prompt}
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            stderr_tail = self._proc.stderr.read()
            raise RuntimeError(f"MLX worker closed its output unexpectedly.\nstderr:\n{stderr_tail}")

        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"MLX worker error: {resp['error']}")
        return resp["text"]

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.terminate()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
