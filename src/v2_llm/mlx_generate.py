"""
mlx_generate.py — long-lived MLX generation worker, run under .venv_mlx's
interpreter (NOT the codenet conda env — mlx-lm's current PyPI release is
incompatible with the transformers version already pinned there; see
mlx_client.py for why this lives in its own subprocess/venv instead of being
downgraded into the shared environment).

Protocol: load the model once (argv[1] = model path/HF repo), then loop
reading one JSON object per line from stdin -> {"system_prompt": ..., "user_prompt": ...}
and writing one JSON object per line to stdout -> {"text": ...} or {"error": ...}.
Prints a single "READY" line to stdout once the model is loaded, before
entering the loop, so the parent process knows loading finished (and can
detect load failures instead of hanging on the first request).
"""
import json
import sys

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MAX_TOKENS = 1500  # KEEP IN SYNC with llm_client.py's MAX_NEW_TOKENS - separate
                    # constant because this file runs under .venv_mlx and can't
                    # import llm_client (pulls in transformers/torch from the
                    # codenet env). Got out of sync once already (stayed at 450
                    # here after llm_client.py was bumped to 1500) and silently
                    # caused 100% parse failure on a verbose model - if you
                    # change one, change both.


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: mlx_generate.py <model_path>"}), flush=True)
        sys.exit(1)

    model_path = sys.argv[1]

    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        # Reported on stdout (not just stderr) so the parent's readiness
        # read gets a parseable line instead of hanging forever.
        print(json.dumps({"error": f"load failed: {e}"}), flush=True)
        sys.exit(1)

    # temp=0.0 -> greedy/argmax decoding, matching LocalHFClient's
    # do_sample=False (reproducible: rerunning shouldn't flip verdicts).
    sampler = make_sampler(temp=0.0)

    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            messages = [
                {"role": "system", "content": req["system_prompt"]},
                {"role": "user", "content": req["user_prompt"]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            text = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=MAX_TOKENS, sampler=sampler, verbose=False,
            )
            print(json.dumps({"text": text}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
