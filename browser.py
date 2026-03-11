import asyncio
import requests
from browser_use_sdk import AsyncBrowserUse

async def main():
    url = "https://www.ip-rs.si/mnenja-zvop-2/videonadzor-v-šoli-1773127489"
    response = requests.get(url)
    response.raise_for_status()
    vsebina_strani = response.text

    prompt = f"""
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
Vsebina spletne strani: {vsebina_strani}
"""

    client = AsyncBrowserUse()
    result = await client.run(prompt)

    print(result.output)

asyncio.run(main())