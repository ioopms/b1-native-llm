"""Command line entry point: generate / run / selftest.

  python -m b1 generate --batch DIR     # one LLM call, then one sandboxed run
  python -m b1 run      --batch DIR --candidate FILE
  python -m b1 selftest --batch DIR     # bundled reference candidate, no API call
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from . import checks, data, prompts, sandbox, score

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def strip_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1]
        code = code.rsplit("```", 1)[0]
    return code.strip()


def call_llm(prompt, model, endpoint, timeout=900):
    key = os.environ.get("deepseek_flash_api")
    if not key:
        raise RuntimeError("set the deepseek_flash_api environment variable")
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.2}
    req = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        method="POST")
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return raw["choices"][0]["message"]["content"], raw, time.time() - start


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_report(out: Path, meta, result, checks_out, extra):
    payload = {
        "batch": str(meta["batch"]),
        "format": meta["format"],
        "n_train": meta["n_train"],
        "n_test": meta["n_test"],
        "n_features": meta["n_features"],
        "n_classes": len(meta["labels"]),
        "metrics": result,
        "checks": checks_out,
        **extra,
    }
    (out / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    md = ["# B1 运行报告", f"批次：`{meta['batch']}`", "",
          f"格式 {meta['format']} / 训练 {meta['n_train']} / 测试 {meta['n_test']} / "
          f"特征 {meta['n_features']} / 类别 {len(meta['labels'])}", "",
          "| 指标 | 结果 |", "|---|---|",
          f"| 准确率（原始标签） | {result['correct_raw']}/{result['n']} = {result['accuracy_raw']:.2%} |",
          f"| 准确率（同义词合并后） | {result['correct']}/{result['n']} = {result['accuracy']:.2%} |",
          f"| 宏平均 F1 | {result['macro_f1']:.6f} |",
          f"| 无效输出 | {result['invalid']} |"]
    if extra.get("generation_seconds") is not None:
        md.append(f"| 生成耗时 | {extra['generation_seconds']:.1f} s |")
    if extra.get("sandbox_seconds") is not None:
        md.append(f"| 沙箱执行 | {extra['sandbox_seconds']:.1f} s |")
    md += ["", "## 一致性检查", ""]
    for key, val in checks_out.items():
        md.append(f"- {key}: {val}")
    if result["errors"]:
        md += ["", "## 错误样本", "", "| 样本 | 真实 | 预测 |", "|---|---|---|"]
        md += [f"| {a} | {b} | {c} |" for a, b, c in result["errors"][:50]]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="b1",
                                     description="B1: one-pass code generation, one sandboxed execution")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "run", "selftest"):
        sp = sub.add_parser(name)
        sp.add_argument("--batch", required=True)
        sp.add_argument("--candidate")
        sp.add_argument("--timeout", type=int, default=300)
        sp.add_argument("--bins", type=int, default=64)
        sp.add_argument("--model", default="deepseek-flash")
        sp.add_argument("--endpoint", default="https://api.deepseek.com/chat/completions")
        sp.add_argument("--no-checks", action="store_true")
        sp.add_argument("--artifacts", default=str(ARTIFACTS))
    args = parser.parse_args(argv)

    batch = Path(args.batch).resolve()
    meta = data.load(batch)
    synonyms_file = batch / "synonyms.json"
    synonyms = json.loads(synonyms_file.read_text(encoding="utf-8")) if synonyms_file.exists() else {}

    prompt = None
    generation = {}
    if args.cmd == "generate":
        prompt = prompts.build_prompt(batch, meta, bins=args.bins, timeout=args.timeout)
        print(f"[B1] prompt {len(prompt)} chars (~{len(prompt)//3} tokens); calling {args.model} ...")
        content, raw, seconds = call_llm(prompt, args.model, args.endpoint)
        code = strip_fences(content)
        generation = {"model": args.model, "generation_seconds": seconds,
                      "usage": raw.get("usage", {})}
        print(f"[B1] generated in {seconds:.1f}s, tokens={generation['usage'].get('total_tokens')}")
    elif args.cmd == "run":
        if not args.candidate:
            parser.error("run requires --candidate")
        code = Path(args.candidate).read_text(encoding="utf-8")
    else:
        ref = ROOT / "candidates" / ("reference_csv.py" if meta["format"] == "csv"
                                     else "reference_json.py")
        code = ref.read_text(encoding="utf-8")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.artifacts) / batch.name / f"{args.cmd}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidate.py").write_text(code, encoding="utf-8")
    if prompt is not None:
        (out / "prompt.txt").write_text(prompt, encoding="utf-8")

    problems = sandbox.check_source(code)
    print(f"[check] source warnings: {problems or 'none'}")

    if meta["format"] == "json":
        box = sandbox.make_json_sandbox(batch, code)
    else:
        box = sandbox.make_csv_sandbox(batch, code)
    proc, elapsed, timed_out = sandbox.run_sandbox(box, args.timeout)
    if proc is not None:
        (out / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (out / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if timed_out:
        print(f"[sandbox] TIMEOUT after {args.timeout}s")
        return 1
    print(f"[sandbox] exit={proc.returncode} elapsed={elapsed:.1f}s")
    if proc.stdout.strip():
        print("--- stdout ---\n" + proc.stdout.strip()[-1500:])
    if proc.stderr.strip():
        print("--- stderr ---\n" + proc.stderr.strip()[-1500:])

    preds = sandbox.read_predictions(box)
    if preds is None:
        print("[B1] candidate produced no valid predictions.json")
        return 1
    (out / "predictions.json").write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")

    result = score.evaluate(meta["truth"], preds, meta["labels"], synonyms)
    print(f"\n[B1] accuracy (raw)     = {result['correct_raw']}/{result['n']} = {result['accuracy_raw']:.2%}")
    print(f"[B1] accuracy (canonical) = {result['correct']}/{result['n']} = {result['accuracy']:.2%}")
    print(f"[B1] macro-F1           = {result['macro_f1']:.6f}")
    for sid, truth, pred in result["errors"][:10]:
        print(f"      {sid}: {truth} -> {pred}")

    checks_out = {}
    if not args.no_checks:
        oi = checks.order_invariance(batch, meta, code, preds, args.timeout)
        ls = checks.label_sensitivity(batch, meta, code, preds, args.timeout)
        checks_out = {"order_invariance": oi, "label_sensitivity": ls}
        print(f"[check] order_invariance: {oi}")
        print(f"[check] label_sensitivity: {ls}")

    extra = {"prompt_sha256": _sha(prompt) if prompt else None,
             "candidate_sha256": _sha(code),
             "source_problems": problems,
             "sandbox_seconds": elapsed,
             **generation}
    write_report(out, meta, result, checks_out, extra)
    print(f"\n[B1] artifacts written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
