import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://www.ip-rs.si/mnenja-zvop-2/"
OUTPUT_FILE = "ip_rs_mnenja_zvop2.json"

PAGE_SIZE = 20
MAX_WORKERS = 12
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "sl-SI,sl;q=0.9,en;q=0.8",
}


def create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session


def fetch_html(session: requests.Session, url: str) -> str:
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def extract_total_results(html: str) -> int:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)

    m = re.search(r"(\d+)\s+najdenih rezultatov", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    m = re.search(r"\b\d+\s*-\s*\d+\s*/\s*(\d+)\b", text)
    if m:
        return int(m.group(1))

    raise RuntimeError("Nisem uspel razbrati skupnega števila rezultatov.")


def build_page_urls(total_results: int) -> List[str]:
    total_pages = math.ceil(total_results / PAGE_SIZE)
    urls = [BASE_URL]
    for page_idx in range(1, total_pages):
        offset = page_idx * PAGE_SIZE
        urls.append(f"{BASE_URL}?offset={offset}")
    return urls


def resolve_ip_link(href: str) -> Optional[str]:
    """
    Razreši href v končni absolutni URL.
    Podpira:
    - direktne poti: /mnenja-zvop-2/...
    - redirect poti: /go?u=%2Fmnenja-zvop-2%2F...
    """
    if not href:
        return None

    full = urljoin(BASE_URL, href.strip())
    parsed = urlparse(full)

    # Direktni link na mnenje
    if parsed.path.startswith("/mnenja-zvop-2/") and parsed.path.rstrip("/") != "/mnenja-zvop-2":
        return full

    # Redirect link /go?u=...
    if parsed.path == "/go":
        qs = parse_qs(parsed.query)
        target = qs.get("u", [None])[0]
        if target:
            target = unquote(target)
            final_url = urljoin("https://www.ip-rs.si", target)
            final_parsed = urlparse(final_url)
            if final_parsed.path.startswith("/mnenja-zvop-2/") and final_parsed.path.rstrip("/") != "/mnenja-zvop-2":
                return final_url

    return None


def is_probable_opinion_anchor(a) -> bool:
    """
    Poskusi izločiti samo povezave na posamezna mnenja.
    Na strani so mnenja prikazana kot naslovi v seznamu rezultatov.
    """
    title = a.get_text(" ", strip=True)
    href = a.get("href", "").strip()

    if not title or len(title) < 3:
        return False

    low = title.lower()
    banned = {
        "naslednja",
        "prejšnja",
        "help",
        "domov",
        "informacijski pooblaščenec",
    }
    if low in banned:
        return False

    resolved = resolve_ip_link(href)
    if not resolved:
        return False

    return True


def extract_opinions_from_page(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, str]] = []

    # Pobiramo vse anchorje, potem filtriramo na mnenja.
    for a in soup.find_all("a", href=True):
        if not is_probable_opinion_anchor(a):
            continue

        title = a.get_text(" ", strip=True)
        link = resolve_ip_link(a["href"])

        if link:
            results.append({
                "title": title,
                "link": link,
            })

    return dedupe_by_link(results)


def dedupe_by_link(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: Set[str] = set()
    out: List[Dict[str, str]] = []
    for item in items:
        link = item["link"]
        if link not in seen:
            seen.add(link)
            out.append(item)
    return out


def validate_results(items: List[Dict[str, str]], expected_total: int) -> None:
    if not items:
        raise RuntimeError("Ni bilo najdenih nobenih rezultatov.")

    for i, item in enumerate(items):
        if "title" not in item or "link" not in item:
            raise RuntimeError(f"Neveljaven zapis #{i}: {item}")
        if not item["title"].strip():
            raise RuntimeError(f"Prazen title pri zapisu #{i}: {item}")
        if not item["link"].startswith("https://"):
            raise RuntimeError(f"Link ni https pri zapisu #{i}: {item}")

    # Ker je na strani javno prikazano skupno število rezultatov, pričakujemo približno toliko zapisov.
    # Če jih je bistveno manj, je parser verjetno napačen.
    if len(items) < int(expected_total * 0.95):
        raise RuntimeError(
            f"Najdenih zapisov je premalo: {len(items)} / pričakovanih {expected_total}. "
            "Parser verjetno ni pravilno zajel vseh mnenj."
        )


def scrape_page(session: requests.Session, url: str, idx: int, total_pages: int) -> List[Dict[str, str]]:
    html = fetch_html(session, url)
    items = extract_opinions_from_page(html)
    print(f"[INFO] Stran {idx}/{total_pages}: najdenih {len(items)} mnenj")
    return items


def main() -> None:
    session = create_session()

    print("[INFO] Pridobivam prvo stran...")
    first_html = fetch_html(session, BASE_URL)
    total_results = extract_total_results(first_html)
    print(f"[INFO] Skupno pričakovanih rezultatov: {total_results}")

    page_urls = build_page_urls(total_results)
    total_pages = len(page_urls)
    print(f"[INFO] Skupno strani za prenos: {total_pages}")

    all_items: List[Dict[str, str]] = []

    # Prvo stran že imamo, zato jo takoj obdelamo.
    first_items = extract_opinions_from_page(first_html)
    print(f"[INFO] Stran 1/{total_pages}: najdenih {len(first_items)} mnenj")
    all_items.extend(first_items)

    # Ostale strani obdelamo paralelno.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(scrape_page, session, url, idx + 1, total_pages): idx + 1
            for idx, url in enumerate(page_urls[1:], start=1)
        }

        for future in as_completed(future_map):
            page_items = future.result()
            all_items.extend(page_items)

    deduped = dedupe_by_link(all_items)
    deduped.sort(key=lambda x: x["link"])

    print(f"[INFO] Po deduplikaciji: {len(deduped)} zapisov")

    # Izpišemo nekaj vzorcev, da se hitro vidi, ali so podatki smiselni.
    print("[DEBUG] Prvih 5 zapisov:")
    for row in deduped[:5]:
        print(" ", row)

    validate_results(deduped, total_results)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"[OK] Shranjeno v {OUTPUT_FILE}: {len(deduped)} zapisov")


if __name__ == "__main__":
    main()