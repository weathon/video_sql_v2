import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import sympy as sp
import wandb


async def main():
    problems = [
        {"id": 1, "kind": "indefinite", "integrand": "x**5 - 3*x**2 + 7"},
        {"id": 2, "kind": "indefinite", "integrand": "sin(x)*cos(x)"},
        {"id": 3, "kind": "indefinite", "integrand": "exp(2*x)*sin(3*x)"},
        {"id": 4, "kind": "indefinite", "integrand": "1/(x**2 + 1)"},
        {"id": 5, "kind": "indefinite", "integrand": "log(x)"},
        {"id": 6, "kind": "indefinite", "integrand": "x*exp(x**2)"},
        {"id": 7, "kind": "indefinite", "integrand": "1/(x*(x + 1))"},
        {"id": 8, "kind": "indefinite", "integrand": "1/sqrt(1 - x**2)"},
        {"id": 9, "kind": "indefinite", "integrand": "x**2*cos(x)"},
        {"id": 10, "kind": "indefinite", "integrand": "1/(x + 2)"},
        {
            "id": 11,
            "kind": "definite",
            "integrand": "x**4*(1 - x)**3",
            "lower": "0",
            "upper": "1",
        },
        {
            "id": 12,
            "kind": "definite",
            "integrand": "exp(-x)",
            "lower": "0",
            "upper": "oo",
        },
        {
            "id": 13,
            "kind": "definite",
            "integrand": "sin(x)",
            "lower": "0",
            "upper": "pi",
        },
        {
            "id": 14,
            "kind": "definite",
            "integrand": "log(x)",
            "lower": "0",
            "upper": "1",
        },
        {
            "id": 15,
            "kind": "definite",
            "integrand": "1/(1 + x**2)",
            "lower": "0",
            "upper": "oo",
        },
        {
            "id": 16,
            "kind": "definite",
            "integrand": "x*sin(x)",
            "lower": "0",
            "upper": "pi",
        },
        {
            "id": 17,
            "kind": "definite",
            "integrand": "sqrt(1 - x**2)",
            "lower": "0",
            "upper": "1",
        },
        {
            "id": 18,
            "kind": "definite",
            "integrand": "exp(x)*cos(x)",
            "lower": "0",
            "upper": "pi",
        },
        {
            "id": 19,
            "kind": "definite",
            "integrand": "1/(sqrt(x)*(1 + x))",
            "lower": "0",
            "upper": "oo",
        },
        {
            "id": 20,
            "kind": "definite",
            "integrand": "x**2*exp(-x)",
            "lower": "0",
            "upper": "oo",
        },
    ]
    root = Path(__file__).resolve().parents[1] / "runs" / "qwen38_integrals"
    run_label = os.environ["RUN_LABEL"]
    run_dir = root / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    problems_path = run_dir / "problems.json"
    if problems_path.exists():
        assert json.loads(problems_path.read_text()) == problems
    else:
        problems_path.write_text(
            json.dumps(problems, ensure_ascii=False, indent=2)
        )

    x = sp.symbols("x", real=True)
    sympy_locals = {
        "x": x,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "pi": sp.pi,
        "E": sp.E,
        "oo": sp.oo,
        "Rational": sp.Rational,
    }
    records = []
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=60.0)
    ) as client:
        models_response = await client.get("http://127.0.0.1:8000/v1/models")
        models_response.raise_for_status()
        model = models_response.json()["data"][0]["id"]
        wandb_run = wandb.init(
            project="video-sql-v2",
            name=f"qwen38-integrals-{run_label}",
            dir=root,
            config={
                "run_label": run_label,
                "model": model,
                "problem_count": len(problems),
                "trials_per_problem": 3,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "reasoning_effort": "xhigh",
                "tensor_parallel_size": 2,
                "gpu_indices": [5, 6],
            },
        )
        for problem in problems:
            integrand = sp.sympify(problem["integrand"], locals=sympy_locals)
            if problem["kind"] == "indefinite":
                expected = sp.integrate(integrand, x)
                question = f"计算不定积分：integrate({problem['integrand']}, x)"
            else:
                lower = sp.sympify(problem["lower"], locals=sympy_locals)
                upper = sp.sympify(problem["upper"], locals=sympy_locals)
                expected = sp.integrate(integrand, (x, lower, upper))
                question = (
                    "计算定积分："
                    f"integrate({problem['integrand']}, (x, "
                    f"{problem['lower']}, {problem['upper']}))"
                )
            for trial in range(3):
                stem = f"problem_{problem['id']:02d}_trial_{trial + 1}"
                raw_path = run_dir / f"{stem}_raw.json"
                verified_path = run_dir / f"{stem}_verified.json"
                if raw_path.exists():
                    response_json = json.loads(raw_path.read_text())
                else:
                    payload = {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "请精确求解下面的积分。你可以推理，但最终答案必须"
                                    "单独放在一行，格式严格为 FINAL: <SymPy表达式>。"
                                    "变量使用 x，可使用 exp、log、sin、cos、sqrt、atan、"
                                    "asin、pi、E、oo、Rational；不要写 + C，不要给小数近似。\n"
                                    f"{question}"
                                ),
                            }
                        ],
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "top_k": 20,
                        "seed": 20260826 + trial,
                        "chat_template_kwargs": {"reasoning_effort": "xhigh"},
                    }
                    print(
                        json.dumps(
                            {
                                "event": "request",
                                "run_label": run_label,
                                "problem_id": problem["id"],
                                "trial": trial + 1,
                                "seed": payload["seed"],
                            }
                        ),
                        flush=True,
                    )
                    request_start = time.monotonic()
                    response = await client.post(
                        "http://127.0.0.1:8000/v1/chat/completions",
                        json=payload,
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    raw_path.write_text(
                        json.dumps(response_json, ensure_ascii=False, indent=2)
                    )
                    print(
                        json.dumps(
                            {
                                "event": "response",
                                "run_label": run_label,
                                "problem_id": problem["id"],
                                "trial": trial + 1,
                                "elapsed_seconds": time.monotonic() - request_start,
                                "finish_reason": response_json["choices"][0][
                                    "finish_reason"
                                ],
                                "completion_tokens": response_json["usage"][
                                    "completion_tokens"
                                ],
                            }
                        ),
                        flush=True,
                    )

                if verified_path.exists():
                    record = json.loads(verified_path.read_text())
                else:
                    choice = response_json["choices"][0]
                    content = choice["message"]["content"]
                    final_lines = [
                        line.strip()
                        for line in content.splitlines()
                        if line.strip().startswith("FINAL:")
                    ]
                    record = {
                        "run_label": run_label,
                        "model": model,
                        "problem": problem,
                        "trial": trial + 1,
                        "seed": 20260826 + trial,
                        "expected": str(expected),
                        "finish_reason": choice["finish_reason"],
                        "usage": response_json["usage"],
                        "content": content,
                        "reasoning": choice["message"]["reasoning_content"],
                        "format_valid": len(final_lines) == 1,
                    }
                    if len(final_lines) != 1:
                        record["answer"] = None
                        record["residual"] = None
                        record["parse_error"] = (
                            f"expected exactly one FINAL line, got {len(final_lines)}"
                        )
                        record["correct"] = False
                    else:
                        answer_text = final_lines[0].removeprefix("FINAL:").strip()
                        record["answer"] = answer_text
                        try:
                            answer = sp.sympify(answer_text, locals=sympy_locals)
                        except (sp.SympifyError, SyntaxError) as error:
                            record["residual"] = None
                            record["parse_error"] = repr(error)
                            record["correct"] = False
                        else:
                            if problem["kind"] == "indefinite":
                                residual = sp.simplify(sp.diff(answer, x) - integrand)
                            else:
                                residual = sp.simplify(answer - expected)
                            record["residual"] = str(residual)
                            record["parse_error"] = None
                            record["correct"] = residual == 0
                    verified_path.write_text(
                        json.dumps(record, ensure_ascii=False, indent=2)
                    )
                records.append(record)
                wandb.log(
                    {
                        "sample/problem_id": problem["id"],
                        "sample/trial": trial + 1,
                        "sample/correct": int(record["correct"]),
                        "sample/format_valid": int(record["format_valid"]),
                        "sample/completion_tokens": record["usage"][
                            "completion_tokens"
                        ],
                    }
                )

        correct_count = sum(int(record["correct"]) for record in records)
        format_valid_count = sum(int(record["format_valid"]) for record in records)
        summary = {
            "run_label": run_label,
            "model": model,
            "sample_count": len(records),
            "correct_count": correct_count,
            "accuracy": correct_count / len(records),
            "format_valid_count": format_valid_count,
            "format_valid_rate": format_valid_count / len(records),
            "per_problem": {
                str(problem["id"]): {
                    "correct_count": sum(
                        int(record["correct"])
                        for record in records
                        if record["problem"]["id"] == problem["id"]
                    ),
                    "trial_count": 3,
                }
                for problem in problems
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )
        wandb.log(
            {
                "result/accuracy": summary["accuracy"],
                "result/correct_count": correct_count,
                "result/format_valid_rate": summary["format_valid_rate"],
            }
        )
        wandb_run.finish()
        print(json.dumps(summary, ensure_ascii=False), flush=True)


asyncio.run(main())
