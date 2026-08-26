import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx
import sympy as sp
import wandb


async def main():
    problems = [
        {"id": 1, "kind": "indefinite", "integrand": "x**4*exp(2*x)"},
        {"id": 2, "kind": "indefinite", "integrand": "x**3*sin(2*x)"},
        {"id": 3, "kind": "indefinite", "integrand": "x**2*log(x)**2"},
        {
            "id": 4,
            "kind": "indefinite",
            "integrand": "x**2/sqrt(x**2 + 1)",
        },
        {"id": 5, "kind": "indefinite", "integrand": "1/(x**4 + 1)"},
        {
            "id": 6,
            "kind": "indefinite",
            "integrand": "(3*x**2 + 5*x + 1)/((x + 1)*(x**2 + 1))",
        },
        {
            "id": 7,
            "kind": "indefinite",
            "integrand": "sin(x)**5*cos(x)**4",
        },
        {
            "id": 8,
            "kind": "indefinite",
            "integrand": "x**3*exp(-x**2)",
        },
        {
            "id": 9,
            "kind": "indefinite",
            "integrand": "atan(x)**2/(1 + x**2)",
        },
        {"id": 10, "kind": "indefinite", "integrand": "log(x)**3/x"},
        {
            "id": 11,
            "kind": "definite",
            "integrand": "log(x)**4",
            "lower": "0",
            "upper": "1",
            "expected": "24",
        },
        {
            "id": 12,
            "kind": "definite",
            "integrand": "x**5/(exp(x) - 1)",
            "lower": "0",
            "upper": "oo",
            "expected": "8*pi**6/63",
        },
        {
            "id": 13,
            "kind": "definite",
            "integrand": "1/(x**4 + 1)",
            "lower": "0",
            "upper": "oo",
            "expected": "sqrt(2)*pi/4",
        },
        {
            "id": 14,
            "kind": "definite",
            "integrand": "log(x)/(1 + x)",
            "lower": "0",
            "upper": "1",
            "expected": "-pi**2/12",
        },
        {
            "id": 15,
            "kind": "definite",
            "integrand": "x**2/(1 + x**2)**2",
            "lower": "0",
            "upper": "oo",
            "expected": "pi/4",
        },
        {
            "id": 16,
            "kind": "definite",
            "integrand": "atan(x)/(1 + x**2)",
            "lower": "0",
            "upper": "1",
            "expected": "pi**2/32",
        },
        {
            "id": 17,
            "kind": "definite",
            "integrand": "x**4*exp(-x**2)",
            "lower": "0",
            "upper": "oo",
            "expected": "3*sqrt(pi)/8",
        },
        {
            "id": 18,
            "kind": "definite",
            "integrand": "log(1 + x)/x",
            "lower": "0",
            "upper": "1",
            "expected": "pi**2/12",
        },
        {
            "id": 19,
            "kind": "definite",
            "integrand": "sqrt(x)/(1 + x)**2",
            "lower": "0",
            "upper": "oo",
            "expected": "pi/2",
        },
        {
            "id": 20,
            "kind": "definite",
            "integrand": "exp(-x)*sin(x)**2",
            "lower": "0",
            "upper": "oo",
            "expected": "Rational(2, 5)",
        },
    ]
    root = Path(__file__).resolve().parents[1] / "runs" / "qwen38_hard_integrals"
    run_label = os.environ["RUN_LABEL"]
    run_dir = root / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    problems_path = run_dir / "problems.json"
    if problems_path.exists():
        assert json.loads(problems_path.read_text()) == problems
    else:
        problems_path.write_text(json.dumps(problems, ensure_ascii=False, indent=2))

    x = sp.symbols("x", real=True)
    sympy_locals = {
        "x": x,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "Abs": sp.Abs,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "asinh": sp.asinh,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "zeta": sp.zeta,
        "polylog": sp.polylog,
        "factorial": sp.factorial,
        "gamma": sp.gamma,
        "pi": sp.pi,
        "E": sp.E,
        "oo": sp.oo,
        "Rational": sp.Rational,
    }
    abs_argument = sp.Wild("abs_argument")
    records = []
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=60.0)
    ) as client:
        models_response = await client.get("http://127.0.0.1:8000/v1/models")
        models_response.raise_for_status()
        model = models_response.json()["data"][0]["id"]
        wandb_run = wandb.init(
            project="video-sql-v2",
            name=f"qwen38-hard-integrals-{run_label}",
            dir=root,
            config={
                "run_label": run_label,
                "model": model,
                "problem_count": len(problems),
                "trials_per_problem": 1,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "reasoning_effort": "xhigh",
                "seed": 20260826,
                "tensor_parallel_size": 2,
                "gpu_indices": [4, 5],
            },
        )
        for problem in problems:
            integrand = sp.sympify(problem["integrand"], locals=sympy_locals)
            if problem["kind"] == "indefinite":
                expected = sp.integrate(integrand, x)
                question = f"计算不定积分：integrate({problem['integrand']}, x)"
            else:
                expected = sp.sympify(problem["expected"], locals=sympy_locals)
                question = (
                    "计算定积分："
                    f"integrate({problem['integrand']}, (x, "
                    f"{problem['lower']}, {problem['upper']}))"
                )

            stem = f"problem_{problem['id']:02d}"
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
                                "asin、asinh、Abs、zeta、polylog、factorial、gamma、"
                                "pi、E、oo、Rational；不要写 + C，"
                                "不要给小数近似。\n"
                                f"{question}"
                            ),
                        }
                    ],
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "seed": 20260826,
                    "chat_template_kwargs": {"reasoning_effort": "xhigh"},
                }
                print(
                    json.dumps(
                        {
                            "event": "request",
                            "run_label": run_label,
                            "problem_id": problem["id"],
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
                if content is None:
                    final_lines = []
                else:
                    final_lines = [
                        line.strip()
                        for line in content.splitlines()
                        if line.strip().startswith("FINAL:")
                    ]
                record = {
                    "run_label": run_label,
                    "model": model,
                    "problem": problem,
                    "seed": 20260826,
                    "expected": str(expected),
                    "finish_reason": choice["finish_reason"],
                    "usage": response_json["usage"],
                    "content": content,
                    "reasoning": choice["message"]["reasoning"],
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
                    identifiers = set(
                        re.findall(r"[A-Za-z_][A-Za-z_0-9]*", answer_text)
                    )
                    unsupported_identifiers = identifiers - sympy_locals.keys()
                    if unsupported_identifiers:
                        record["residual"] = None
                        record["parse_error"] = (
                            "unsupported identifiers: "
                            f"{sorted(unsupported_identifiers)}"
                        )
                        record["correct"] = False
                    else:
                        try:
                            answer = sp.sympify(answer_text, locals=sympy_locals)
                        except (sp.SympifyError, SyntaxError, TypeError) as error:
                            record["residual"] = None
                            record["parse_error"] = repr(error)
                            record["correct"] = False
                        else:
                            if not isinstance(answer, sp.Expr):
                                record["residual"] = None
                                record["parse_error"] = (
                                    "answer is not a scalar SymPy expression: "
                                    f"{type(answer).__name__}"
                                )
                                record["correct"] = False
                            else:
                                normalized_answer = answer.replace(
                                    sp.log(sp.Abs(abs_argument)),
                                    sp.log(abs_argument),
                                )
                                if problem["kind"] == "indefinite":
                                    residual = sp.simplify(
                                        sp.diff(normalized_answer, x) - integrand
                                    )
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
                str(record["problem"]["id"]): {
                    "correct": record["correct"],
                    "format_valid": record["format_valid"],
                }
                for record in records
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
