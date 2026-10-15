# Semantic AI Router 🚀

A local Flask dashboard demonstrating semantic caching, model routing, and simulated cost analytics.

## ✨ Features

- **Semantic caching:** Matches similar questions and reuses cached responses.
- **Model routing:** Demonstrates routing questions to different model tiers.
- **Performance analytics:** Tracks requests, cache hits, cache misses, and latency.
- **Cost simulation:** Estimates illustrative costs and savings.
- **Request history:** Displays previous questions and cache results.
- **Dark dashboard:** A modern interface with navy, purple, and blue accents.
- **Local demo:** Runs on your computer without requiring a paid AI API.

## 🛠️ Technologies

- Python
- Flask
- Sentence Transformers
- NumPy
- HTML and CSS

## ▶️ Run locally

1. Clone or download this repository.
2. Install the required packages.
3. Start the Flask application.

```bash
python -m pip install flask sentence-transformers numpy
python app.py
```

4. Open `http://127.0.0.1:5000` in your browser.

## 📊 Important notes

This is a prototype. Responses and model-routing decisions are simulated. Cost and savings values are illustrative, not actual API charges or measured production savings. Analytics and cache data are stored in memory and reset when the application restarts.

## 👩‍💻 Author

Created as a personal learning project to explore AI infrastructure concepts, semantic caching, and cost-aware routing.
