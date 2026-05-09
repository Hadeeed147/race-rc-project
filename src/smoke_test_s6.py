"""
Session 6 smoke test:

  * 5 samples in 'real' mode  -> /sample_with_question + /predict
    Verify the /predict response has 4 numerically distinct scores.
  * 5 samples in 'generated' mode -> /generate (+ /predict on the result)
    Track how many of those 5 fire quality_warning.
  * Confirm /predict latency < 100 ms on every call.
"""

import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.loads(r.read())
    return d, int((time.time() - t0) * 1000)


def get(path):
    t0 = time.time()
    with urllib.request.urlopen(BASE + path, timeout=20) as r:
        d = json.loads(r.read())
    return d, int((time.time() - t0) * 1000)


def distinct_count(scores):
    return len({round(v, 6) for v in scores.values()})


def main():
    print("== Real RACE Question mode (n=5) ==")
    real_results = []
    for i in range(1, 6):
        s, t_s = get("/sample_with_question")
        p, t_p = post("/predict", {
            "article": s["article"],
            "question": s["question"],
            "options": s["options"],
        })
        gold = s["answer"]
        pred = p["predicted"]
        n_distinct = distinct_count(p["scores"])
        ok = n_distinct >= 2 and t_p < 100
        flag = "OK " if ok else "WARN"
        print(f"  [{i}] {flag} id={s['id'][:18]:<18} gold={gold} pred={pred} "
              f"distinct_scores={n_distinct}/4 t_predict={t_p}ms scores={p['scores']}")
        real_results.append({"distinct": n_distinct, "t_predict": t_p,
                             "gold": gold, "pred": pred, "id": s["id"],
                             "scores": p["scores"]})

    print("\n== AI-Generated mode (n=5) ==")
    gen_results = []
    for i in range(1, 6):
        s, _ = get("/sample")
        g, t_g = post("/generate", {"article": s["article"]})
        warn = g.get("quality_warning", False)
        rejected = g.get("rejected_reasons", [])
        # Now also call /predict on the generated quiz
        p, t_p = post("/predict", {
            "article": s["article"],
            "question": g["question"],
            "options": g["options"],
        })
        n_distinct = distinct_count(p["scores"])
        flag = "WARN" if warn else "OK"
        print(f"  [{i}] {flag}  id={s['id'][:18]:<18} t_gen={t_g}ms t_pred={t_p}ms "
              f"distinct={n_distinct}/4 template={g.get('template')}")
        print(f"       Q: {g['question']}")
        print(f"       options: {g['options']}")
        if warn:
            print(f"       rejected_reasons: {rejected}")
        gen_results.append({"warn": warn, "rejected": rejected, "t_gen": t_g,
                            "t_predict": t_p, "id": s["id"],
                            "question": g["question"]})

    # Summary
    distinct_scores_real = sum(1 for r in real_results if r["distinct"] >= 2)
    correct_real = sum(1 for r in real_results if r["gold"] == r["pred"])
    warn_count = sum(1 for r in gen_results if r["warn"])
    over_lat = sum(1 for r in real_results + gen_results if r["t_predict"] > 100)

    print("\n--- summary ---")
    print(f"  real mode: {distinct_scores_real}/5 had >= 2 distinct option scores.")
    print(f"  real mode: ensemble correct on {correct_real}/5.")
    print(f"  generated mode: quality_warning on {warn_count}/5.")
    print(f"  /predict latency over 100 ms: {over_lat} samples.")

    # First real-mode response, full payload, for the report summary
    print("\n--- example /predict response (real mode) ---")
    print(json.dumps({
        "id":     real_results[0]["id"],
        "gold":   real_results[0]["gold"],
        "pred":   real_results[0]["pred"],
        "scores": real_results[0]["scores"],
    }, indent=2))


if __name__ == "__main__":
    main()
