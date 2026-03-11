import os
import json
import requests
import re

API_URL = "https://llm.505labs.ai/v1/chat/completions"
API_KEY = "sk-T1SbLsMCxCJ8uk1q-F3xxg"
MODEL = "global.anthropic.claude-sonnet-4-6"
DATA_FOLDER = "./json_datoteke"

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "x-api-key": API_KEY,
}

def load_json_files(folder_path):
    responses = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                odgovor = data.get("povzetek", {}).get("odgovor")
                if odgovor:
                    responses.append(f"{data.get('naslov')}: {odgovor}")
    return responses

def check_global_contradictions(responses):
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
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0,
        "max_tokens": 800,
        "stream": False
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    result = response.json()
    output = result['choices'][0]['message']['content'].strip()

    # Poskusimo izvleči JSON iz outputa (če je model dodal dodatno besedilo)
    try:
        # Poiščemo prvi JSON objekt v tekstu
        match = re.search(r'\{.*\}', output, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            # Če JSON ni najden, vrnemo default
            return {"protislovje": False, "komentar": "Ni protislovij"}
    except json.JSONDecodeError:
        return {"protislovje": False, "komentar": "Ni protislovij"}

if __name__ == "__main__":
    responses = load_json_files(DATA_FOLDER)
    if responses:
        print(json.dumps(check_global_contradictions(responses)))