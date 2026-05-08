"""Quick smoke test: hit each of the 5 backend endpoints and print results."""
import json
import urllib.request

BASE = "http://127.0.0.1:8000"
ART = (
    "Honey bees are insects that live in colonies of up to 60,000 individuals. "
    "A single colony has one queen, hundreds of male drones, and tens of "
    "thousands of female worker bees. The workers forage for nectar and pollen "
    "and produce the wax used to build the comb. The queen lays all the eggs "
    "in the colony, sometimes more than 1,500 a day during peak season."
)


def post(path, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())


def main():
    print("== /generate ==")
    g = post("/generate", {"article": ART})
    print(json.dumps(g, indent=2)[:500])

    print("\n== /predict ==")
    p = post("/predict", {
        "article": ART,
        "question": "Who lays the eggs in a honey bee colony?",
        "options": {
            "A": "The drones",
            "B": "The worker bees",
            "C": "The queen",
            "D": "The foragers",
        },
    })
    print(json.dumps(p, indent=2))

    print("\n== /distractors ==")
    d = post("/distractors", {"article": ART, "correct_answer": "the queen"})
    print(json.dumps(d, indent=2))

    print("\n== /hints ==")
    h = post("/hints", {"article": ART, "question": "Who lays the eggs?"})
    print(json.dumps(h, indent=2))

    print("\n== /analytics ==")
    a = get("/analytics")
    print(f"keys: {list(a.keys())}")
    if "model_a" in a:
        print(f"model_a entries: {len(a['model_a'])}")
    if "model_b" in a:
        print(f"model_b: {list(a['model_b'].keys())}")

    print("\nAll endpoints reachable. ✓")


if __name__ == "__main__":
    main()
