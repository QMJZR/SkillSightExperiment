import argparse
import csv
import json
import os
import re
import sys
import time

import requests


def load_prompt(prompt_dir):
    system = open(os.path.join(prompt_dir, "system_v1.txt"), encoding="utf-8").read().strip()
    template = open(os.path.join(prompt_dir, "user_template_v1.txt"), encoding="utf-8").read().strip()
    return system, template


def build_user(template, row, max_chars):
    values = {
        "query": row.get("query", ""),
        "reference_answer": (row.get("reference_answer", "") or "")[:1000],
        "skill_name": row.get("skill_name", ""),
        "skill_description": row.get("skill_description", ""),
        "skill_snippet": (row.get("skill_snippet", "") or "")[:max_chars],
    }
    return template.format(**values)


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start : end + 1])


def normalize(parsed):
    out = {
        "requirements": parsed.get("requirements", []),
        "semantic": int(parsed.get("semantic", -1)),
        "coverage": int(parsed.get("coverage", -1)),
        "functional_equivalent": bool(parsed.get("functional_equivalent", False)),
        "invalid_reason": str(parsed.get("invalid_reason", "") or ""),
        "evidence": str(parsed.get("evidence", "") or ""),
        "confidence": int(parsed.get("confidence", 1)),
    }
    if out["semantic"] not in (0, 1, 2):
        raise ValueError(f"bad semantic: {out['semantic']}")
    if out["coverage"] not in (0, 1, 2):
        raise ValueError(f"bad coverage: {out['coverage']}")
    return out


def call_chat(cfg, system, user, temperature=0.0, max_tokens=600):
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(f"missing env {cfg['api_key_env']}")
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 400 and "response_format" in payload:
                payload.pop("response_format", None)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            if attempt == 2:
                raise
            print(f"  retry {attempt + 1}: {exc}", flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="盲标 CSV")
    ap.add_argument("--prompt-dir", default=os.path.join(os.path.dirname(__file__)))
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "providers.json"))
    ap.add_argument("--providers", default="deepseek,qwen,zhipu,openai")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pair-ids", default="")
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    system, template = load_prompt(args.prompt_dir)
    config = json.load(open(args.config, encoding="utf-8"))
    rows = list(csv.DictReader(open(args.input, encoding="utf-8")))
    wanted = {x.strip() for x in args.pair_ids.split(",") if x.strip()}
    if wanted:
        rows = [r for r in rows if r["pair_id"] in wanted]
    if args.limit:
        rows = rows[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    for provider in providers:
        if provider not in config:
            print(f"[skip] unknown provider {provider}")
            continue
        cfg = config[provider]
        if not os.environ.get(cfg["api_key_env"]):
            print(f"[skip] {provider}: {cfg['api_key_env']} not set")
            continue
        out_path = os.path.join(args.out_dir, f"{provider}.jsonl")
        done = set()
        if os.path.exists(out_path):
            for line in open(out_path, encoding="utf-8"):
                try:
                    done.add(json.loads(line)["pair_id"])
                except Exception:
                    pass
        todo = [r for r in rows if r["pair_id"] not in done]
        print(f"[{provider}] model={cfg['model']} todo={len(todo)} done={len(done)}")
        if args.dry_run:
            if todo:
                print(build_user(template, todo[0], args.max_chars)[:1200])
            continue
        with open(out_path, "a", encoding="utf-8") as fh:
            for i, row in enumerate(todo, start=1):
                user = build_user(template, row, args.max_chars)
                rec = {"pair_id": row["pair_id"], "provider": provider, "model": cfg["model"]}
                try:
                    raw = call_chat(cfg, system, user)
                    rec["parsed"] = normalize(parse_json(raw))
                    rec["raw"] = raw
                    rec["error"] = None
                except Exception as exc:
                    rec["parsed"] = None
                    rec["raw"] = None
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 20 == 0:
                    print(f"  [{provider}] {i}/{len(todo)}", flush=True)
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
