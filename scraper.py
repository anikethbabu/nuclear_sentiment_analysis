import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ====== CONFIG ======
GUARDIAN_API_KEY = os.getenv('API_KEY')
TARGET_PER_SOURCE = 100

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Shorter sleep for faster scraping
SLEEP_TIME = 0.3

# ====== DATABASE ======
conn = sqlite3.connect("nuclear.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    title TEXT,
    url TEXT UNIQUE,
    content TEXT
)
""")

def save_article(source, title, url, content):
    try:
        cursor.execute(
            "INSERT OR IGNORE INTO articles (source, title, url, content) VALUES (?, ?, ?, ?)",
            (source, title, url, content)
        )
        conn.commit()
    except:
        pass

def count_source(source):
    cursor.execute("SELECT COUNT(*) FROM articles WHERE source=?", (source,))
    return cursor.fetchone()[0]

# ====== GUARDIAN API ======
def scrape_guardian():
    print("Scraping Guardian...")
    page = 1

    while count_source("guardian") < TARGET_PER_SOURCE:
        url = f"https://content.guardianapis.com/search?q=nuclear&page={page}&api-key={GUARDIAN_API_KEY}&show-fields=bodyText"
        res = requests.get(url).json()

        results = res["response"]["results"]

        if not results:
            break

        for article in results:
            title = article["webTitle"]
            link = article["webUrl"]
            content = article["fields"].get("bodyText", "")

            save_article("guardian", title, link, content)

        page += 1
        time.sleep(1)

# ====== GREENPEACE ======
def scrape_greenpeace():
    print("Scraping Greenpeace...")
    page = 1

    while count_source("greenpeace") < TARGET_PER_SOURCE:
        url = f"https://www.greenpeace.org/international/tag/nuclear/page/{page}/"
        soup = BeautifulSoup(requests.get(url, headers=HEADERS).text, "html.parser")

        links = [a["href"] for a in soup.find_all("a", href=True) if "/story/" in a["href"]]

        if not links:
            break

        for link in links:
            try:
                page_soup = BeautifulSoup(requests.get(link, headers=HEADERS).text, "html.parser")
                title = page_soup.title.text
                content = " ".join([p.text for p in page_soup.find_all("p")])

                save_article("greenpeace", title, link, content)
                time.sleep(SLEEP_TIME)
            except:
                continue

        page += 1

# ====== WORLD NUCLEAR NEWS ======
def scrape_wnn():
    print("Scraping World Nuclear News...")
    page = 1

    while count_source("wnn") < TARGET_PER_SOURCE:
        url = f"https://www.world-nuclear-news.org/search?search=nuclear&page={page}"
        soup = BeautifulSoup(requests.get(url, headers=HEADERS).text, "html.parser")

        links = [a["href"] for a in soup.find_all("a", href=True) if "/articles/" in a["href"]]

        if not links:
            break

        for link in links:
            try:
                full_url = f"https://www.world-nuclear-news.org{link}"
                page_soup = BeautifulSoup(requests.get(full_url, headers=HEADERS).text, "html.parser")
                title = page_soup.title.text
                content = " ".join([p.text for p in page_soup.find_all("p")])

                save_article("wnn", title, full_url, content)
                time.sleep(SLEEP_TIME)
            except:
                continue

        page += 1

# ====== WORLD NUCLEAR ASSOCIATION ======
def scrape_wna():
    print("Scraping WNA...")
    
    url = "https://world-nuclear.org/information-library"
    soup = BeautifulSoup(requests.get(url, headers=HEADERS).text, "html.parser")

    links = [a["href"] for a in soup.find_all("a", href=True) if "/information-library/" in a["href"] and len(a["href"]) > 20]

    for link in links:
        if count_source("wna") >= TARGET_PER_SOURCE:
            break
            
        try:
            full_url = f"https://world-nuclear.org{link}"
            page_soup = BeautifulSoup(requests.get(full_url, headers=HEADERS).text, "html.parser")
            title = page_soup.title.text if page_soup.title else "No Title"
            content = " ".join([p.text for p in page_soup.find_all("p")])

            save_article("wna", title, full_url, content)
            time.sleep(SLEEP_TIME)
        except:
            continue

# ====== BEYOND NUCLEAR ======
def scrape_beyond_nuclear():
    print("Scraping Beyond Nuclear...")
    page = 1

    while count_source("beyond_nuclear") < TARGET_PER_SOURCE:
        url = f"https://beyondnuclear.org/nuclear-power/news/page/{page}/"
        soup = BeautifulSoup(requests.get(url, headers=HEADERS).text, "html.parser")

        # Known navigation URLs to exclude
        exclude_urls = {
            "https://beyondnuclear.org/",
            "https://beyondnuclear.org/take-action/",
            "https://beyondnuclear.org/about/",
            "https://beyondnuclear.org/events/",
            "https://beyondnuclear.org/climate-crisis/",
            "https://beyondnuclear.org/health-impacts/",
            "https://beyondnuclear.org/nuclear-power/",
            "https://beyondnuclear.org/nuclear-weapons/",
            "https://beyondnuclear.org/publications/",
            "https://beyondnuclear.org/radioactive-waste/"
        }

        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if (href.startswith("https://beyondnuclear.org/") and 
                href not in exclude_urls and
                not any(skip in href for skip in ["facebook", "twitter", "youtube", "instagram", "constantcontact", "#"])):
                if href not in links:
                    links.append(href)

        if not links:
            break

        for link in links:
            try:
                page_soup = BeautifulSoup(requests.get(link, headers=HEADERS).text, "html.parser")
                title = page_soup.title.text if page_soup.title else "No Title"
                content = " ".join([p.text for p in page_soup.find_all("p")])

                save_article("beyond_nuclear", title, link, content)
                time.sleep(SLEEP_TIME)
            except:
                continue

        page += 1

# ====== RUN ======
if __name__ == "__main__":
    # scrape_guardian()
    # scrape_greenpeace()
    scrape_wnn()
    scrape_wna()
    scrape_beyond_nuclear()

    print("\nFinal counts:")
    for s in ["guardian", "greenpeace", "wnn", "wna", "beyond_nuclear"]:
        print(s, count_source(s))
