#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List
from urllib import request

API_URL = "https://llm.505labs.ai/v1/chat/completions"
MODEL = "global.anthropic.claude-sonnet-4-6"
DEFAULT_DATA_FOLDER = "./json_datoteke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check contradictions across extracted opinion summaries")
    parser.add_argument("--data-folder", default=DEFAULT_DATA_FOLDER, help="Folder with extracted JSON files")
    return parser.parse_args()


def get_api_key() -> str:
    api_key = os.environ.get("LABS_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set LABS_API_KEY or API_KEY env variable.")
    return api_key


def load_json_files(folder_path: str) -> List[str]:
    responses: List[str] = []
    path = Path(folder_path)
    if not path.exists():
        return responses

    for filename in sorted(path.iterdir()):
        if filename.suffix.lower() != ".json":
            continue
        with filename.open("r", encoding="utf-8") as f:
            data = json.load(f)
            odgovor = data.get("povzetek", {}).get("odgovor")
            if odgovor:
                responses.append(f"{data.get('naslov', filename.stem)}: {odgovor}")
    return responses


def check_global_contradictions(responses: List[str], api_key: str) -> Dict[str, Any]:
    prompt_text = (
        "Preveri vse spodnje odgovore skupaj glede možnih protislovij med njimi.\n"
        "Vrni SAMO JSON v obliki:\n"
        '{"protislovje": false, "komentar": "Ni protislovij"}\n'
        "Če obstajajo protislovja, JSON: "
        '{"protislovje": true, "komentar": "...kratko strnjeno pojasnilo konfliktov..."}\n\n'
    )
    for idx, odgovor in enumerate(responses, 1):
        prompt_text += f"{idx}. {odgovor}\n"

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0,
        "max_tokens": 800,
        "stream": False,
    }

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        API_URL,
        data=body,
        headers={
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": api_key,
        },
        method="POST",
    )
    with request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    output = result["choices"][0]["message"]["content"].strip()
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"protislovje": False, "komentar": "Ni protislovij"}


if __name__ == "__main__":
    args = parse_args()
    responses = load_json_files(args.data_folder)
    if responses:
        print(json.dumps(check_global_contradictions(responses, get_api_key()), ensure_ascii=False))
    else:
        print(json.dumps({"protislovje": False, "komentar": "Ni podatkov za analizo"}, ensure_ascii=False))
