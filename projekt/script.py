import asyncio
import json
import re
import requests
from pathlib import Path

from fastapi import FastAPI
from bs4 import BeautifulSoup
from json_repair import repair_json
from browser_use_sdk import AsyncBrowserUse

app = FastAPI()

INPUT_FILE = "rrf_results.json"
OUTPUT_DIR = Path("./json_datoteke")
OUTPUT_DIR.mkdir(exist_ok=True)

client = AsyncBrowserUse()


def clean_html(html):

    soup = BeautifulSoup(html, "html.parser")

    article = soup.find("article")

    if article:
        text = article.get_text(" ", strip=True)
    else:
        text = soup.get_text(" ", strip=True)

    return text[:15000]


def parse_llm_json(output):

    cleaned = re.sub(r"```json|```", "", output).strip()

    try:
        return json.loads(cleaned)
    except:
        repaired = repair_json(cleaned)
        return json.loads(repaired)


async def run_llm(prompt, retries=3):

    for i in range(retries):

        result = await client.run(
            prompt,
            max_tokens=2500
        )

        try:
            return parse_llm_json(result.output)
        except:

            if i == retries - 1:
                raise

            print("Retry extraction...")


async def extract_page(item, index):

    url = item["link"]

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        vsebina_strani = clean_html(response.text)

        prompt = f"""
Si pravni strokovnjak za področje varstva osebnih podatkov. Iz vsebine spletne strani mnenja Informacijskega pooblaščenca (IP-RS) ekstrahiraj naslednje podatke in jih vrni izključno kot veljaven JSON brez kakršnega koli dodatnega besedila.

Pravila za ekstrakcijo:

naslov: Naslov mnenja. 
URL: url naslov tega mnenja
številka: Uradna številka mnenja.
kategorije: Seznam kategorij kot JSON array.
datum: Datum mnenja v obliki DD.MM.LLLL.
pravna_podlaga: Seznam vseh pravnih podlag kot JSON array. Izpusti vse pravne podlage, ki so navedene v uvodnem odstavku mnenja, ki se začne z besedami »Na podlagi informacij, ki ste nam jih posredovali, vam v nadaljevanju skladno s …«. Te pravne podlage so podlaga za izdajo mnenja samega (tj. pristojnost IP) in ne vsebinska pravna podlaga obravnavanega vprašanja. Tipično gre za: 5. točko prvega odstavka 55. člena ZVOP-2, 58. člen Splošne uredbe in 2. člen ZInfP.
povzetek: Objekt s tremi ključi:
dejansko_stanje: Kratek opis dejanskega stanja. Namesto generičnih izrazov kot »upravljavec« uporabi konkretno ime subjekta (npr. »osnovna šola«, »podjetje«). Namesto »pobudnik« uporabi »zaprositelj za mnenje«; sicer se tej besedi izogibaj. Izpusti vse sklicevanje na postopek izdaje mnenja ali pristojnosti IP.
pravno_vprašanje: Kratko in jedrnato pravno vprašanje v eni povedi.
odgovor: Jedrnat pravni odgovor z uporabo pravne terminologije. Izpusti vse stavke, ki se nanašajo na omejitve neobvezujočega mnenja IP ali pristojnosti inšpekcijskega postopka.
Vsebina spletne strani:

Vsebina strani: {vsebina_strani}
"""

        parsed = await run_llm(prompt)

        parsed["url"] = url

        stevilka = parsed.get('stevilka')
        if not stevilka or stevilka.strip() == "":
            stevilka = f"file_{index+1}"

        filename = OUTPUT_DIR / f"{stevilka.replace('/','_')}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)

        return parsed


    except Exception:

        print(f"Napaka pri parsiranju: {url}")
        return None


async def run_pipeline():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)["top_results"]

    tasks = [extract_page(item, idx) for idx,item in enumerate(results)]

    extracted = await asyncio.gather(*tasks)

    extracted = [e for e in extracted if e]

    final_json = {
        "generalno_menje_o_zadevi": "",
        "table": extracted
    }

    with open("final_output.json", "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    return final_json


@app.get("/mock")
async def mock():
    result = await run_pipeline()

    return result