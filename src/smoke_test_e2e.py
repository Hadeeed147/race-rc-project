"""
Session 4 end-to-end smoke test.

For 10 random val samples, walk the full backend flow:
  GET  /sample              -> pull a real RACE row
  POST /generate            -> generate a question from that article
  POST /predict             -> ensemble predicts on the gold-options form
  POST /hints               -> 3 graduated hints
  GET  /analytics           -> dashboard payload (once, at the end)

Reports latency stats and any failures. All latencies must be < 10 000 ms.
"""

import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
N = 10


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


def main():
    print(f"Hitting {BASE} -- {N} samples\n")
    rows = []
    for i in range(1, N + 1):
        try:
            samp, t_samp = get("/sample")
            article = samp["article"]
            gen, t_gen = post("/generate", {"article": article})
            pred, t_pred = post("/predict", {
                "article": article,
                "question": samp["question"],
                "options": samp["options"],
            })
            hints, t_hints = post("/hints", {"article": article, "question": samp["question"]})

            ok = (
                pred["predicted"] in {"A", "B", "C", "D"}
                and len(hints["hints"]) == 3
            )
            rows.append({
                "i": i, "id": samp["id"],
                "ok": ok,
                "gold": samp["answer"], "pred": pred["predicted"],
                "t_sample": t_samp, "t_generate": t_gen,
                "t_predict": t_pred, "t_hints": t_hints,
                "latency_predict_ms": pred["latency_ms"],
                "latency_hints_ms": hints["latency_ms"],
                "template": gen.get("template"),
            })
            mark = "OK" if ok else "FAIL"
            print(f"  [{i:2}] {mark}  id={samp['id'][:18]:<18}  gold={samp['answer']} pred={pred['predicted']}  "
                  f"sample={t_samp}ms gen={t_gen}ms pred={t_pred}ms hints={t_hints}ms")
        except Exception as e:
            print(f"  [{i:2}] EXC  {e}")
            rows.append({"i": i, "ok": False, "error": str(e)})

    # Latency summary
    lat = [r["t_predict"] for r in rows if r.get("t_predict") is not None]
    if lat:
        print(f"\n/predict latency: mean={sum(lat)/len(lat):.0f}ms  max={max(lat)}ms")
    h_lat = [r["t_hints"] for r in rows if r.get("t_hints") is not None]
    if h_lat:
        print(f"/hints latency:   mean={sum(h_lat)/len(h_lat):.0f}ms  max={max(h_lat)}ms")
    g_lat = [r["t_generate"] for r in rows if r.get("t_generate") is not None]
    if g_lat:
        print(f"/generate latency:mean={sum(g_lat)/len(g_lat):.0f}ms  max={max(g_lat)}ms")
    s_lat = [r["t_sample"] for r in rows if r.get("t_sample") is not None]
    if s_lat:
        print(f"/sample latency:  mean={sum(s_lat)/len(s_lat):.0f}ms  max={max(s_lat)}ms")

    correct = sum(1 for r in rows if r.get("ok") and r.get("gold") == r.get("pred"))
    total_ok = sum(1 for r in rows if r.get("ok"))
    print(f"\nEnsemble correct (per-question argmax): {correct}/{total_ok}")

    # /analytics
    a, t_a = get("/analytics")
    print(f"\n/analytics returned keys={list(a.keys())} ({t_a}ms)")

    # Latency budget check
    over = [r for r in rows if r.get("ok") and (
        r["t_predict"] > 10_000 or r["t_hints"] > 10_000 or r["t_generate"] > 10_000
    )]
    if over:
        print(f"\nWARN: {len(over)} samples exceeded 10 000 ms budget.")
    else:
        print("\nAll endpoints under 10 000 ms budget.")


if __name__ == "__main__":
    main()
