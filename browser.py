#!/usr/bin/env python3
import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict
from urllib import request

PROMPT_TEMPLATE = """
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

Vsebina spletne strani:
{vsebina_strani}
""".strip()


class ExtractionError(RuntimeError):
    pass


def extract_json_from_output(output: str) -> Dict[str, Any]:
    text = output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ExtractionError("Model output does not contain a JSON object")
        return json.loads(match.group())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured JSON from an IP-RS opinion URL")
    parser.add_argument("--url", required=True, help="IP-RS opinion URL")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds for fetching the page")
    return parser.parse_args()


async def run_extraction(url: str, output_path: Path, timeout: int) -> None:
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    prompt = PROMPT_TEMPLATE.format(vsebina_strani=html)
    from browser_use_sdk import AsyncBrowserUse

    client = AsyncBrowserUse()
    result = await client.run(prompt)

    parsed = extract_json_from_output(result.output)
    parsed["source_link"] = url

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    asyncio.run(run_extraction(args.url, Path(args.output), args.timeout))


if __name__ == "__main__":
    main()
