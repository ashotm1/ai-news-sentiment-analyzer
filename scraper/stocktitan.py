import requests
from bs4 import BeautifulSoup

FINBERT_URL = "http://127.0.0.1:5000/analyze"


def analyze_sentiment(texts):
    res = requests.post(FINBERT_URL, json={"text": texts})
    return res.json()


def scrape(limit=10):
    url = "https://www.stocktitan.net/news/live.html"
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    feed = soup.find("div", attrs={"role": "feed"})
    if not feed:
        return []

    box = feed.find("div", class_=lambda c: c and "d-flex py-2" in c)

    articles = []
    count = 0

    while box and count < limit:
        ticker_div = box.find("div", class_="news-list-tickers")
        ticker = ticker_div.find("span").get_text(strip=True) if ticker_div else "N/A"

        title_div = box.find("div", {"name": "title"})
        link = title_div.find("a") if title_div else None

        title = link.get_text(strip=True) if link else "No title"
        url_link = link["href"] if link else "N/A"

        tags_div = box.find("div", class_="news-list-tags")
        tags = [t.get_text(strip=True) for t in tags_div.find_all("span")] if tags_div else []

        articles.append({
            "ticker": ticker,
            "title": title,
            "url": url_link,
            "tags": tags
        })

        count += 1
        box = box.find_next("div", class_=lambda c: c and "d-flex py-2" in c)

    # sentiment batch call
    titles = [a["title"] for a in articles]
    sentiments = analyze_sentiment(titles)

    for i, a in enumerate(articles):
        s = sentiments[i]
        a["sentiment"] = s["label"]
        a["score"] = float(s["score"])

    return articles