# Hiter test
# python -c "import json; d=json.load(open('ip_rs_mnenja_zvop2_enriched.json', encoding='utf-8')); print(len(d)); print(d[0]['title']); print(d[0]['date']); print(d[0]['case_number']); print(d[0]['categories'])"

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://www.ip-rs.si/mnenja-zvop-2/"
OUTPUT_FILE = "ip_rs_mnenja_zvop2_enriched.json"

# BASE_URL = "https://www.ip-rs.si/mnenja-zvop-2/?id=13176&asId=as0&search=&oseba=Za+fizi%C4%8Dne+osebe+oz.+posameznike&pubfromdate=&pubtodate=&sub=Iskanje"
# OUTPUT_FILE = "ip_rs_mnenja_zvop2_enriched_fizicne_posamezniki.json"

PAGE_SIZE = 20
MAX_WORKERS_LIST = 12
MAX_WORKERS_DETAIL = 12
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
    adapter = HTTPAdapter(max_retries=retry, pool_connections=30, pool_maxsize=30)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session


def fetch_html(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def extract_total_results(html: str) -> int:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)

    match = re.search(r"(\d+)\s+najdenih rezultatov", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"\b\d+\s*-\s*\d+\s*/\s*(\d+)\b", text)
    if match:
        return int(match.group(1))

    raise RuntimeError("Nisem uspel razbrati skupnega števila rezultatov.")


def build_page_urls(total_results: int) -> List[str]:
    total_pages = math.ceil(total_results / PAGE_SIZE)
    urls = [BASE_URL]
    for page_idx in range(1, total_pages):
        offset = page_idx * PAGE_SIZE
        urls.append(f"{BASE_URL}?offset={offset}")
        # urls.append(f"{BASE_URL}&offset={offset}")
    return urls


def resolve_ip_link(href: str) -> Optional[str]:
    if not href:
        return None

    full = urljoin(BASE_URL, href.strip())
    parsed = urlparse(full)

    if parsed.path.startswith("/mnenja-zvop-2/") and parsed.path.rstrip("/") != "/mnenja-zvop-2":
        return full

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


def dedupe_by_link(items: List[Dict]) -> List[Dict]:
    seen: Set[str] = set()
    out: List[Dict] = []
    for item in items:
        link = item["link"]
        if link not in seen:
            seen.add(link)
            out.append(item)
    return out


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_listing_line(line: str) -> Optional[Dict]:
    """
    Iz tekstovne vrstice pobere:
    datum + naslov + številka + kategorije
    Primer:
    09.03.2026 Videonadzor v šoli 07121-1/2025/1598 Šolstvo, Video in avdio nadzor
    """
    line = normalize_spaces(line)
    if not line:
        return None

    # Datum na začetku
    date_match = re.match(r"^(\d{2}\.\d{2}\.\d{4})\s+(.+)$", line)
    if not date_match:
        return None

    date_str = date_match.group(1)
    rest = date_match.group(2)

    # Številka zadeve je ponavadi vzorec s številkami, vezaji in slashi
    # Primeri:
    # 07121-1/2025/1598
    # 07120-1/2026/91
    # 5424-1/2025/7
    case_match = re.search(r"\b(\d{3,}-[\d/]+)\b", rest)
    if not case_match:
        return None

    case_number = case_match.group(1)
    title = normalize_spaces(rest[:case_match.start()])
    categories_text = normalize_spaces(rest[case_match.end():])

    if not title:
        return None

    categories = []
    if categories_text:
        categories = [normalize_spaces(x) for x in categories_text.split(",") if normalize_spaces(x)]

    return {
        "date": date_str,
        "title": title,
        "case_number": case_number,
        "categories": categories,
    }


def extract_listing_items_from_page(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict] = []

    # Na strani iščemo vse linke na posamezna mnenja
    for a in soup.find_all("a", href=True):
        resolved = resolve_ip_link(a["href"])
        if not resolved:
            continue

        title = normalize_spaces(a.get_text(" ", strip=True))
        if len(title) < 3:
            continue

        # Poskusi dobiti celotno vrstico / blok, ki vsebuje datum, naslov, številko in kategorije
        parent_text_candidates = []

        if a.parent:
            parent_text_candidates.append(normalize_spaces(a.parent.get_text(" ", strip=True)))

        if a.parent and a.parent.parent:
            parent_text_candidates.append(normalize_spaces(a.parent.parent.get_text(" ", strip=True)))

        if isinstance(a, Tag):
            parent_text_candidates.append(normalize_spaces(a.get_text(" ", strip=True)))

        parsed_meta = None
        for candidate in parent_text_candidates:
            parsed = parse_listing_line(candidate)
            if parsed and parsed["title"] == title:
                parsed_meta = parsed
                break

        # Fallback: če title ne matcha povsem, vzemi prvi smiselni parse
        if not parsed_meta:
            for candidate in parent_text_candidates:
                parsed = parse_listing_line(candidate)
                if parsed:
                    parsed_meta = parsed
                    break

        if parsed_meta:
            items.append({
                "title": parsed_meta["title"],
                "link": resolved,
                "date": parsed_meta["date"],
                "case_number": parsed_meta["case_number"],
                "categories": parsed_meta["categories"],
            })

    return dedupe_by_link(items)


def extract_detail_fields(html: str, url: str) -> Dict:
    soup = BeautifulSoup(html, "html.parser")

    title = None
    h1 = soup.find("h1")
    if h1:
        title = normalize_spaces(h1.get_text(" ", strip=True))

    text = soup.get_text("\n", strip=True)

    date_match = re.search(r"Datum:\s*(\d{2}\.\d{2}\.\d{4})", text)
    case_match = re.search(r"Številka:\s*([0-9][0-9A-Za-z\-/]+)", text)
    cat_match = re.search(r"Kategorije:\s*(.+)", text)

    date_str = date_match.group(1) if date_match else None
    case_number = case_match.group(1) if case_match else None

    categories = []
    if cat_match:
        first_line = normalize_spaces(cat_match.group(1).split("\n")[0])
        categories = [normalize_spaces(x) for x in first_line.split(",") if normalize_spaces(x)]

    # Poskusi dobiti prvi smiselni odstavek vsebine po glavi
    paragraphs = []
    for p in soup.find_all(["p"]):
        p_text = normalize_spaces(p.get_text(" ", strip=True))
        if p_text:
            paragraphs.append(p_text)

    excerpt = None
    for p in paragraphs:
        if (
            not p.startswith("Datum:")
            and not p.startswith("Številka:")
            and not p.startswith("Kategorije:")
            and len(p) > 80
        ):
            excerpt = p
            break

    # Polno besedilo: očisti vrhnjo navigacijo, a za robustnost vzamemo vsebinski del okoli H1 naprej
    full_text = None
    if h1:
        full_lines = []
        started = False
        for line in soup.get_text("\n").splitlines():
            line = normalize_spaces(line)
            if not line:
                continue
            if line == title:
                started = True
            if started:
                full_lines.append(line)

        if full_lines:
            full_text = "\n".join(full_lines)

    # Kdo je pripravil mnenje
    prepared_by = None
    prepared_match = re.search(r"Pripravila:\s*(.+)", text)
    if prepared_match:
        prepared_by = normalize_spaces(prepared_match.group(1))

    # Intended audience: na detail strani ga običajno ni eksplicitno videti
    intended_audience = None

    return {
        "title": title,
        "link": url,
        "date": date_str,
        "case_number": case_number,
        "categories": categories,
        "excerpt": excerpt,
        "full_text": full_text,
        "prepared_by": prepared_by,
        "intended_audience": intended_audience,
    }


def merge_listing_and_detail(listing_item: Dict, detail_item: Dict) -> Dict:
    return {
        "title": detail_item.get("title") or listing_item.get("title"),
        "link": listing_item["link"],
        "date": detail_item.get("date") or listing_item.get("date"),
        "case_number": detail_item.get("case_number") or listing_item.get("case_number"),
        "categories": detail_item.get("categories") or listing_item.get("categories") or [],
        "excerpt": detail_item.get("excerpt"),
        "full_text": detail_item.get("full_text"),
        "prepared_by": detail_item.get("prepared_by"),
        "intended_audience": detail_item.get("intended_audience"),
    }


def scrape_listing_page(session: requests.Session, url: str, idx: int, total_pages: int) -> List[Dict]:
    html = fetch_html(session, url)
    items = extract_listing_items_from_page(html)
    print(f"[INFO] Listing stran {idx}/{total_pages}: {len(items)} zapisov")
    return items


def scrape_detail_page(session: requests.Session, listing_item: Dict, idx: int, total_items: int) -> Dict:
    html = fetch_html(session, listing_item["link"])
    detail = extract_detail_fields(html, listing_item["link"])
    merged = merge_listing_and_detail(listing_item, detail)
    print(f"[INFO] Detail {idx}/{total_items}: {merged['title']}")
    return merged


def validate_results(items: List[Dict], expected_total: int) -> None:
    if not items:
        raise RuntimeError("Ni bilo najdenih nobenih rezultatov.")

    for i, item in enumerate(items):
        if not item.get("title"):
            raise RuntimeError(f"Prazen title pri zapisu #{i}")
        if not item.get("link", "").startswith("https://"):
            raise RuntimeError(f"Neveljaven link pri zapisu #{i}: {item.get('link')}")
        if not item.get("date"):
            raise RuntimeError(f"Manjka date pri zapisu #{i}: {item.get('title')}")
        if not item.get("case_number"):
            raise RuntimeError(f"Manjka case_number pri zapisu #{i}: {item.get('title')}")
        if not isinstance(item.get("categories"), list):
            raise RuntimeError(f"categories ni list pri zapisu #{i}: {item.get('title')}")

    if len(items) < int(expected_total * 0.95):
        raise RuntimeError(
            f"Najdenih zapisov je premalo: {len(items)} / pričakovanih {expected_total}. "
            "Parser verjetno ni pravilno zajel vseh mnenj."
        )


def main() -> None:
    session = create_session()

    print("[INFO] Pridobivam prvo stran...")
    first_html = fetch_html(session, BASE_URL)
    total_results = extract_total_results(first_html)
    print(f"[INFO] Skupno pričakovanih rezultatov: {total_results}")

    page_urls = build_page_urls(total_results)
    total_pages = len(page_urls)
    print(f"[INFO] Skupno listing strani: {total_pages}")

    listing_items: List[Dict] = []

    # Prva stran
    first_items = extract_listing_items_from_page(first_html)
    print(f"[INFO] Listing stran 1/{total_pages}: {len(first_items)} zapisov")
    listing_items.extend(first_items)

    # Ostale listing strani
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_LIST) as executor:
        future_map = {
            executor.submit(scrape_listing_page, session, url, idx + 1, total_pages): idx + 1
            for idx, url in enumerate(page_urls[1:], start=1)
        }
        for future in as_completed(future_map):
            page_items = future.result()
            listing_items.extend(page_items)

    listing_items = dedupe_by_link(listing_items)
    listing_items.sort(key=lambda x: x["link"])
    print(f"[INFO] Po deduplikaciji listing zapisov: {len(listing_items)}")

    if not listing_items:
        raise RuntimeError("Listing scraper ni našel nobenega mnenja.")

    # Detail scraping
    enriched_items: List[Dict] = []
    total_items = len(listing_items)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DETAIL) as executor:
        future_map = {
            executor.submit(scrape_detail_page, session, item, idx + 1, total_items): idx + 1
            for idx, item in enumerate(listing_items)
        }
        for future in as_completed(future_map):
            enriched_items.append(future.result())

    enriched_items = dedupe_by_link(enriched_items)
    enriched_items.sort(key=lambda x: x["link"])
    print(f"[INFO] Po detail obdelavi in deduplikaciji: {len(enriched_items)} zapisov")

    print("[DEBUG] Prvih 3 zapisov:")
    for row in enriched_items[:3]:
        print(json.dumps(row, ensure_ascii=False, indent=2)[:1000])

    validate_results(enriched_items, total_results)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(enriched_items, f, ensure_ascii=False, indent=2)

    print(f"[OK] Shranjeno v {OUTPUT_FILE}: {len(enriched_items)} zapisov")


if __name__ == "__main__":
    main()