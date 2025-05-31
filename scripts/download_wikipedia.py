# scripts/download_wikipedia.py

import os
import json
import wikipedia

# Set language to English
wikipedia.set_lang("en")

# Topics to fetch from Wikipedia
topics = [
    "Artificial intelligence",
    "Machine learning",
    "Natural language processing",
    "Deep learning",
    "Neural networks",
    "Computer vision",
    "Reinforcement learning",
    "Support vector machine",
    "Decision trees",
    "K-means clustering"
]

# Output directory
output_dir = "data/raw/wikipedia"
os.makedirs(output_dir, exist_ok=True)

# Fetch and save summaries
def fetch_and_save_articles():
    for topic in topics:
        try:
            summary = wikipedia.summary(topic)
            article = {
                "title": topic,
                "summary": summary
            }
            filename = os.path.join(output_dir, f"{topic.replace(' ', '_').lower()}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(article, f, indent=4)
            print(f"✅ Saved: {topic}")
        except Exception as e:
            print(f"❌ Failed to fetch {topic}: {e}")

if __name__ == "__main__":
    fetch_and_save_articles()
