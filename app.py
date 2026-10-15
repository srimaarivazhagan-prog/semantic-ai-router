
from flask import Flask, request, render_template_string
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
import time

app = Flask(__name__)

# -----------------------------
# Semantic Cache and History
# -----------------------------

cache = {}
request_history = []

# -----------------------------
# Analytics
# -----------------------------

stats = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "small_model": 0,
    "large_model": 0,
    "total_cost": 0.0,
    "cost_savings": 0.0,
    "cache_savings": 0.0,
    "last_latency": 0.0
}

# Simulated costs, not real API prices
SMALL_MODEL_COST = 0.001
LARGE_MODEL_COST = 0.005
SIMILARITY_THRESHOLD = 0.70

# Local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def similarity(embedding1, embedding2):
    denominator = norm(embedding1) * norm(embedding2)

    if denominator == 0:
        return 0.0

    return float(dot(embedding1, embedding2) / denominator)


def choose_model(question):
    simple_words = {
        "what", "who", "when", "where", "define"
    }

    words = question.lower().split()

    if len(words) <= 8 and any(
        word.strip("?!.,:;") in simple_words
        for word in words
    ):
        return "small-model"

    return "large-model"


# -----------------------------
# Dashboard Route
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    cache_status = ""
    selected_model = ""
    last_question = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if question:
            last_question = question
            start_time = time.perf_counter()

            stats["total_requests"] += 1

            question_embedding = model.encode(question)
            selected_model = choose_model(question)

            matched_data = None

            # Find a semantically similar cached question
            for cached_question, data in cache.items():
                score = similarity(
                    question_embedding,
                    data["embedding"]
                )

                if score >= SIMILARITY_THRESHOLD:
                    matched_data = data
                    break

            if matched_data is not None:
                # Cache HIT
                answer = matched_data["answer"]
                selected_model = matched_data["model"]
                cache_status = "HIT"

                stats["cache_hits"] += 1

                # Estimated avoided cost, assuming a large-model call
                stats["cache_savings"] += LARGE_MODEL_COST

            else:
                # Cache MISS: simulated response
                cache_status = "MISS"
                stats["cache_misses"] += 1

                answer = (
                    "Demo response: Your question was processed "
                    "by the local Semantic AI Router. "
                    "Connect a language model later if you want "
                    "real AI-generated answers. Question: "
                    + question
                )

                if selected_model == "small-model":
                    stats["small_model"] += 1
                    stats["total_cost"] += SMALL_MODEL_COST

                    stats["cost_savings"] += (
                        LARGE_MODEL_COST - SMALL_MODEL_COST
                    )
                else:
                    stats["large_model"] += 1
                    stats["total_cost"] += LARGE_MODEL_COST

                cache[question] = {
                    "answer": answer,
                    "embedding": question_embedding,
                    "model": selected_model
                }

            # Measure local processing time
            latency = time.perf_counter() - start_time
            stats["last_latency"] = latency

            request_history.append({
                "question": question,
                "model": selected_model,
                "cache": cache_status,
                "latency": latency
            })

    cache_hit_rate = (
        stats["cache_hits"] / stats["total_requests"] * 100
        if stats["total_requests"] else 0.0
    )

    total_savings = (
        stats["cost_savings"] + stats["cache_savings"]
    )

    return render_template_string(
        PAGE,
        stats=stats,
        answer=answer,
        cache_status=cache_status,
        selected_model=selected_model,
        last_question=last_question,
        cache_hit_rate=cache_hit_rate,
        total_savings=total_savings,
        history=list(reversed(request_history))
    )


# -----------------------------
# Dark Navy Dashboard UI
# -----------------------------

PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Semantic AI Router | Dashboard</title>

<style>
:root {
    color-scheme: dark;
    --bg: #080b18;
    --panel: #11172a;
    --panel-light: #151d35;
    --border: #27314d;
    --text: #eef2ff;
    --muted: #929fbd;
    --purple: #a78bfa;
    --blue: #60a5fa;
    --green: #34d399;
    --red: #fb7185;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background:
        radial-gradient(ellipse at 15% 0%, #25205b 0%, transparent 38%),
        radial-gradient(ellipse at 90% 15%, #102b57 0%, transparent 34%),
        var(--bg);
    color: var(--text);
    font-family: Inter, "Segoe UI", Arial, sans-serif;
    min-height: 100vh;
}

button, input {
    font: inherit;
}

.layout {
    display: flex;
    min-height: 100vh;
}

.sidebar {
    width: 245px;
    flex-shrink: 0;
    padding: 30px 20px;
    background: rgba(9, 13, 29, 0.88);
    border-right: 1px solid var(--border);
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 44px;
}

.brand-icon {
    width: 44px;
    height: 44px;
    display: grid;
    place-items: center;
    border-radius: 14px;
    font-size: 23px;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    box-shadow: 0 0 28px #7049ed55;
}

.brand-name {
    font-size: 15px;
    font-weight: 750;
    line-height: 1.4;
}

.brand-sub {
    font-size: 11px;
    color: var(--muted);
    margin-top: 3px;
}

.nav-label {
    color: #7180a3;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.7px;
    margin: 25px 10px 12px;
}

.nav-item {
    display: block;
    padding: 12px;
    border-radius: 10px;
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 6px;
}

.nav-item.active {
    color: #fff;
    background: linear-gradient(90deg, #7542e51f, #3b82f61c);
    border: 1px solid #8060e344;
}

.side-note {
    margin-top: 42px;
    padding: 15px;
    border: 1px solid var(--border);
    border-radius: 13px;
    background: #11172a;
    font-size: 12px;
    color: var(--muted);
    line-height: 1.6;
}

.online-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    background: var(--green);
    border-radius: 50%;
    margin-right: 7px;
    box-shadow: 0 0 10px #34d39988;
}

.main {
    flex: 1;
    min-width: 0;
    padding: 34px clamp(18px, 4vw, 48px);
    max-width: 1600px;
}

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 32px;
}

.eyebrow {
    color: var(--purple);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 9px;
}

h1 {
    font-size: clamp(25px, 3vw, 35px);
    letter-spacing: -1.2px;
    margin: 0;
}

.subtitle {
    color: var(--muted);
    font-size: 13px;
    margin-top: 9px;
    line-height: 1.6;
}

.status-pill {
    white-space: nowrap;
    padding: 10px 13px;
    border-radius: 999px;
    border: 1px solid #245f4a;
    color: #6ee7b7;
    background: #0c332b80;
    font-size: 11px;
}

.section-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin: 28px 0 16px;
}

.section-heading h2 {
    font-size: 16px;
    margin: 0;
    letter-spacing: -.2px;
}

.section-heading span {
    color: var(--muted);
    font-size: 11px;
}

.metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
}

.card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(145deg, #151c32, #101629);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 21px;
    min-width: 0;
    box-shadow: 0 12px 35px #00000015;
    transition: transform .2s, border-color .2s;
}

.card:hover {
    transform: translateY(-3px);
    border-color: #59649b;
}

.card::after {
    content: "";
    position: absolute;
    top: -55px;
    right: -45px;
    width: 110px;
    height: 110px;
    border-radius: 50%;
    background: var(--glow, #8b5cf6);
    filter: blur(55px);
    opacity: .15;
    pointer-events: none;
}

.card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.5;
}

.metric-icon {
    width: 37px;
    height: 37px;
    border-radius: 11px;
    display: grid;
    place-items: center;
    font-size: 17px;
    background: var(--icon-bg, #8b5cf620);
    color: var(--accent, var(--purple));
}

.metric-value {
    font-size: clamp(25px, 3vw, 32px);
    font-weight: 750;
    letter-spacing: -1px;
    margin-top: 19px;
    overflow-wrap: anywhere;
}

.metric-foot {
    margin-top: 8px;
    color: #7e8cab;
    font-size: 11px;
}

.purple {
    --accent: #b9a2ff;
    --glow: #8b5cf6;
    --icon-bg: #8b5cf620;
}

.blue {
    --accent: #7cbaff;
    --glow: #3b82f6;
    --icon-bg: #3b82f620;
}

.green {
    --accent: #6ee7b7;
    --glow: #10b981;
    --icon-bg: #10b98120;
}

.pink {
    --accent: #f9a8d4;
    --glow: #ec4899;
    --icon-bg: #ec489920;
}

.cyan {
    --accent: #67e8f9;
    --glow: #06b6d4;
    --icon-bg: #06b6d420;
}

.question-panel {
    background: linear-gradient(145deg, #171c38, #11172a);
    border: 1px solid #393663;
    border-radius: 18px;
    padding: clamp(20px, 3vw, 30px);
    box-shadow: 0 12px 45px #00000022;
}

.panel-title {
    font-size: 16px;
    font-weight: 700;
    margin: 0 0 7px;
}

.panel-description {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.6;
    margin-bottom: 20px;
}

.question-form {
    display: flex;
    gap: 10px;
}

.question-input {
    flex: 1;
    min-width: 0;
    padding: 15px 16px;
    color: var(--text);
    background: #090e20;
    border: 1px solid #303b5c;
    border-radius: 11px;
    outline: none;
}

.question-input::placeholder {
    color: #7180a3;
}

.question-input:focus {
    border-color: #8b70ed;
    box-shadow: 0 0 0 3px #8b5cf622;
}

.ask-button {
    padding: 0 23px;
    border: 0;
    border-radius: 11px;
    background: linear-gradient(115deg, #8b5cf6, #3b82f6);
    color: white;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 6px 20px #6548d533;
    transition: filter .2s, transform .2s;
}

.ask-button:hover {
    filter: brightness(1.13);
    transform: translateY(-1px);
}

.response {
    margin-top: 20px;
    padding: 19px;
    background: #0b1022;
    border: 1px solid #283452;
    border-radius: 13px;
    overflow-wrap: anywhere;
}

.response-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 13px;
}

.response-header strong {
    font-size: 13px;
}

.response-text {
    color: #c4cee5;
    font-size: 13px;
    line-height: 1.8;
    white-space: pre-wrap;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 9px;
    border-radius: 7px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .6px;
}

.badge-hit {
    background: #0c382d;
    color: #6ee7b7;
    border: 1px solid #22664e;
}

.badge-miss {
    background: #332750;
    color: #c4b5fd;
    border: 1px solid #604a91;
}

.badge-neutral {
    background: #202a43;
    color: #a8b7d6;
    border: 1px solid #344260;
}

.history-panel {
    border: 1px solid var(--border);
    border-radius: 16px;
    background: #11172a;
    overflow: hidden;
}

.table-scroll {
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
    min-width: 600px;
}

thead {
    background: #171f35;
}

th {
    padding: 15px 18px;
    color: #8e9dbd;
    text-align: left;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    white-space: nowrap;
}

td {
    padding: 16px 18px;
    border-top: 1px solid #222c44;
    font-size: 12px;
    color: #c8d2e8;
}

tbody tr:hover {
    background: #1a2340;
}

.question-cell {
    max-width: 360px;
    overflow-wrap: anywhere;
    line-height: 1.6;
}

.model-name {
    color: #b7a1ff;
    white-space: nowrap;
}

.latency {
    color: #7cbaff;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}

.empty-state {
    padding: 35px 20px;
    text-align: center;
    color: var(--muted);
    font-size: 12px;
}

.footer {
    display: flex;
    justify-content: space-between;
    gap: 15px;
    flex-wrap: wrap;
    padding: 24px 0 8px;
    color: #7180a3;
    font-size: 11px;
}

@media (max-width: 1050px) {
    .sidebar {
        width: 205px;
    }

    .metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 720px) {
    .layout {
        display: block;
    }

    .sidebar {
        width: 100%;
        padding: 15px 20px;
        border-right: 0;
        border-bottom: 1px solid var(--border);
    }

    .brand {
        margin-bottom: 0;
    }

    .sidebar .nav-label,
    .sidebar .nav-item,
    .side-note {
        display: none;
    }

    .main {
        padding: 24px 16px;
    }

    .topbar {
        align-items: flex-start;
    }

    .status-pill {
        font-size: 10px;
        padding: 8px;
    }

    .metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
    }

    .card {
        padding: 15px;
    }

    .metric-value {
        font-size: 25px;
    }

    .metric-icon {
        width: 31px;
        height: 31px;
    }

    .question-form {
        flex-direction: column;
    }

    .question-input {
        width: 100%;
    }

    .ask-button {
        padding: 14px;
    }
}

@media (max-width: 390px) {
    .metrics {
        grid-template-columns: 1fr;
    }

    .status-pill {
        white-space: normal;
    }
}
</style>
</head>

<body>
<div class="layout">

    <aside class="sidebar">
        <div class="brand">
            <div class="brand-icon">✳</div>
            <div>
                <div class="brand-name">Semantic AI Router</div>
                <div class="brand-sub">INTELLIGENCE LAYER</div>
            </div>
        </div>

        <div class="nav-label">WORKSPACE</div>
        <div class="nav-item active">▦ &nbsp; Overview</div>
        <div class="nav-item">◈ &nbsp; Semantic Cache</div>
        <div class="nav-item">⇄ &nbsp; Model Routing</div>

        <div class="nav-label">MONITORING</div>
        <div class="nav-item">◷ &nbsp; Performance</div>
        <div class="nav-item">▤ &nbsp; Request History</div>

        <div class="side-note">
            <div style="color:#eef2ff;font-weight:700;margin-bottom:7px;">
                <span class="online-dot"></span>Local demo active
            </div>
            Running on your computer.<br>
            No paid AI API required.
        </div>
    </aside>

    <main class="main">
        <header class="topbar">
            <div>
                <div class="eyebrow">AI INFRASTRUCTURE / OVERVIEW</div>
                <h1>System Dashboard</h1>
                <div class="subtitle">
                    Monitor semantic caching, routing efficiency,
                    and request performance.
                </div>
            </div>
            <div class="status-pill">
                <span class="online-dot"></span> SYSTEM READY
            </div>
        </header>

        <section class="question-panel">
            <h2 class="panel-title">Test your AI router</h2>
            <div class="panel-description">
                Enter a question to test semantic cache matching
                and simulated model routing.
            </div>

            <form class="question-form" method="POST">
                <input
                    class="question-input"
                    type="text"
                    name="question"
                    placeholder="Try: What is Python?"
                    value="{{ last_question }}"
                    maxlength="2000"
                    required
                >
                <button class="ask-button" type="submit">
                    Run request &nbsp; →
                </button>
            </form>

            {% if answer %}
            <div class="response">
                <div class="response-header">
                    <strong>Latest response</strong>
                    {% if cache_status == "HIT" %}
                    <span class="badge badge-hit">● CACHE HIT</span>
                    {% else %}
                    <span class="badge badge-miss">● CACHE MISS</span>
                    {% endif %}
                </div>

                <div class="response-text">{{ answer }}</div>

                <div style="display:flex;gap:9px;flex-wrap:wrap;margin-top:16px;">
                    <span class="badge badge-neutral">
                        MODEL: {{ selected_model }}
                    </span>
                    <span class="badge badge-neutral">
                        LATENCY: {{ "%.3f"|format(stats["last_latency"]) }} s
                    </span>
                </div>
            </div>
            {% endif %}
        </section>

        <div class="section-heading">
            <h2>Performance overview</h2>
            <span>Session analytics</span>
        </div>

        <section class="metrics">
            <div class="card purple">
                <div class="card-top">
                    <div class="metric-label">Total requests</div>
                    <div class="metric-icon">↗</div>
                </div>
                <div class="metric-value">{{ stats["total_requests"] }}</div>
                <div class="metric-foot">Requests processed this session</div>
            </div>

            <div class="card blue">
                <div class="card-top">
                    <div class="metric-label">Last response time</div>
                    <div class="metric-icon">◷</div>
                </div>
                <div class="metric-value">
                    {{ "%.3f"|format(stats["last_latency"]) }}<span style="font-size:15px"> s</span>
                </div>
                <div class="metric-foot">Local processing time</div>
            </div>

            <div class="card cyan">
                <div class="card-top">
                    <div class="metric-label">Cache hit rate</div>
                    <div class="metric-icon">◎</div>
                </div>
                <div class="metric-value">{{ "%.1f"|format(cache_hit_rate) }}%</div>
                <div class="metric-foot">Requests served from cache</div>
            </div>

            <div class="card green">
                <div class="card-top">
                    <div class="metric-label">Cache hits</div>
                    <div class="metric-icon">✓</div>
                </div>
                <div class="metric-value">{{ stats["cache_hits"] }}</div>
                <div class="metric-foot">Similar cached questions found</div>
            </div>

            <div class="card pink">
                <div class="card-top">
                    <div class="metric-label">Cache misses</div>
                    <div class="metric-icon">⌕</div>
                </div>
                <div class="metric-value">{{ stats["cache_misses"] }}</div>
                <div class="metric-foot">New demo responses generated</div>
            </div>

            <div class="card purple">
                <div class="card-top">
                    <div class="metric-label">Small-model route</div>
                    <div class="metric-icon">↘</div>
                </div>
                <div class="metric-value">{{ stats["small_model"] }}</div>
                <div class="metric-foot">Simulated cheaper-model requests</div>
            </div>

            <div class="card blue">
                <div class="card-top">
                    <div class="metric-label">Large-model route</div>
                    <div class="metric-icon">↗</div>
                </div>
                <div class="metric-value">{{ stats["large_model"] }}</div>
                <div class="metric-foot">Simulated higher-cost requests</div>
            </div>

            <div class="card cyan">
                <div class="card-top">
                    <div class="metric-label">Estimated cost</div>
                    <div class="metric-icon">$</div>
                </div>
                <div class="metric-value">${{ "%.3f"|format(stats["total_cost"]) }}</div>
                <div class="metric-foot">Based on demo cost assumptions</div>
            </div>

            <div class="card green">
                <div class="card-top">
                    <div class="metric-label">Estimated savings</div>
                    <div class="metric-icon">✧</div>
                </div>
                <div class="metric-value">${{ "%.3f"|format(total_savings) }}</div>
                <div class="metric-foot">Simulated routing + cache savings</div>
            </div>
        </section>

        <div class="section-heading">
            <h2>Request history</h2>
            <span>{{ history|length }} recorded requests</span>
        </div>

        <section class="history-panel">
            <div class="table-scroll">
                <table>
                    <thead>
                        <tr>
                            <th>Question</th>
                            <th>Selected model</th>
                            <th>Cache status</th>
                            <th>Latency</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for item in history %}
                        <tr>
                            <td class="question-cell">{{ item["question"] }}</td>
                            <td class="model-name">{{ item["model"] }}</td>
                            <td>
                                {% if item["cache"] == "HIT" %}
                                <span class="badge badge-hit">HIT</span>
                                {% else %}
                                <span class="badge badge-miss">MISS</span>
                                {% endif %}
                            </td>
                            <td class="latency">
                                {{ "%.3f"|format(item["latency"]) }} s
                            </td>
                        </tr>
                        {% else %}
                        <tr>
                            <td colspan="4">
                                <div class="empty-state">
                                    No requests yet. Ask your first question above
                                    to populate the history.
                                </div>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </section>

        <footer class="footer">
            <span>SEMANTIC AI ROUTER · LOCAL PROTOTYPE</span>
            <span>Semantic cache · Routing · Analytics</span>
        </footer>
    </main>
</div>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)
