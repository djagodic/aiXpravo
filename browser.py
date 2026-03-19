import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def extract_basic_fields(html: str, url: str) -> Dict[str, Any]:
    """Offline fallback extraction that does not require Browser Use API key."""
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text("\n", strip=True)

    title = ""
    h1 = soup.find("h1")
    if h1:
        title = normalize_spaces(h1.get_text(" ", strip=True))
    if not title and soup.title:
        title = normalize_spaces(soup.title.get_text(" ", strip=True))

    case_number_match = re.search(r"Številka:\s*([0-9][0-9A-Za-z\-/]+)", page_text)
    date_match = re.search(r"Datum:\s*(\d{2}\.\d{2}\.\d{4})", page_text)
    categories_match = re.search(r"Kategorije:\s*(.+)", page_text)

    categories: List[str] = []
    if categories_match:
        first_line = normalize_spaces(categories_match.group(1).split("\n")[0])
        categories = [normalize_spaces(part) for part in first_line.split(",") if normalize_spaces(part)]

    legal_basis_candidates = re.findall(r"\b\d+\.\s*člen[^\n.;]{0,120}", page_text, flags=re.IGNORECASE)
    legal_basis = [
        item
        for item in dict.fromkeys(normalize_spaces(candidate) for candidate in legal_basis_candidates)
        if "2. člen Zakona o informacijskem pooblaščencu" not in item
    ]

    paragraphs = [
        normalize_spaces(p.get_text(" ", strip=True))
        for p in soup.find_all("p")
        if normalize_spaces(p.get_text(" ", strip=True))
    ]
    content_paragraphs = [
        p
        for p in paragraphs
        if not p.startswith("Datum:") and not p.startswith("Številka:") and not p.startswith("Kategorije:")
    ]

    dejansko_stanje = content_paragraphs[0] if content_paragraphs else ""
    pravno_vprasanje = content_paragraphs[1] if len(content_paragraphs) > 1 else ""
    odgovor = content_paragraphs[2] if len(content_paragraphs) > 2 else (content_paragraphs[-1] if content_paragraphs else "")

    return {
        "url": url,
        "naslov": title,
        "številka": case_number_match.group(1) if case_number_match else "",
        "kategorije": categories,
        "datum": date_match.group(1) if date_match else "",
        "pravna_podlaga": legal_basis,
        "povzetek": {
            "dejansko_stanje": dejansko_stanje,
            "pravno_vprašanje": pravno_vprasanje,
            "odgovor": odgovor,
        },
    }


async def extract_with_browser_use(prompt: str, api_key: str) -> str:
    from browser_use_sdk import AsyncBrowserUse

    client = AsyncBrowserUse(api_key=api_key)
    result = await client.run(prompt)
    return result.output


def build_prompt(page_html: str) -> str:
    return f"""
Si pravni strokovnjak za področje varstva osebnih podatkov. Iz vsebine spletne strani mnenja Informacijskega pooblaščenca (IP-RS) ekstrahiraj naslednje podatke in jih vrni izključno kot veljaven JSON brez kakršnega koli dodatnega besedila.

Pravila za ekstrakcijo:

naslov: Naslov mnenja.
številka: Uradna številka mnenja.
kategorije: Seznam kategorij kot JSON array.
datum: Datum mnenja v obliki DD.MM.LLLL.
pravna_podlaga: Seznam vseh pravnih podlag kot JSON array. Izpusti 2. člen Zakona o informacijskem pooblaščencu (ZInfP), saj je ta vedno prisoten.
povzetek: Objekt s tremi ključi:
  dejansko_stanje: Kratek opis dejanskega stanja. Namesto generičnih izrazov kot „upravljavec" uporabi konkretno ime subjekta (npr. „osnovna šola", „podjetje"). Namesto „pobudnik" uporabi „zaprositelj za mnenje"; sicer se tej besedi izogibaj. Izpusti vse sklicevanje na postopek izdaje mnenja ali pristojnosti IP.
  pravno_vprašanje: Kratko in jedrnato pravno vprašanje v eni povedi.
  odgovor: Jedrnat pravni odgovor z uporabo pravne terminologije. Izpusti vse stavke, ki se nanašajo na omejitve neobvezujočega mnenja IP ali pristojnosti inšpekcijskega postopka.
Vsebina spletne strani: {page_html}
"""


def parse_json_output(raw_output: str) -> Dict[str, Any]:
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


async def run_extraction(url: str, output_path: Path, timeout: int, api_key: str | None = None) -> Dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    html = response.text

    selected_api_key = api_key or os.getenv("BROWSER_USE_API_KEY")
    if selected_api_key:
        prompt = build_prompt(html)
        raw = await extract_with_browser_use(prompt, selected_api_key)
        data = parse_json_output(raw)
        if isinstance(data, dict):
            data.setdefault("url", url)
        else:
            raise ValueError("Browser Use ni vrnil JSON objekta.")
    else:
        print("[browser.py] BROWSER_USE_API_KEY ni nastavljen; uporabljam fallback ekstrakcijo brez LLM.")
        data = extract_basic_fields(html, url)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract IP-RS opinion data to JSON.")
    parser.add_argument("--url", required=True, help="URL of the IP-RS opinion page")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    parser.add_argument("--api-key", default=None, help="Optional Browser Use API key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_extraction(args.url, Path(args.output), args.timeout, args.api_key))


if __name__ == "__main__":
    main()
