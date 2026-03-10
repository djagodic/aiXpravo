# aiXpravo
Ai x Pravo hekathon on FMF

Workflow:
crawler (ip-rs)
      ↓
embedding generator (prednost je: hitrejše iskanje, šparanje na tokenih, semantična podobnost, manj halucinacij)
      ↓
vector database
      ↓
cosine similarity search (PCA graf)
      ↓
top 10 mnenj (z Broser Usom bomo iz 10 najbližjih extractal ključne podatke)
      ↓
LLM povzetek (Broswer Use) <=> Tax-Fin-Lex (keywords, info laws …)
     ↓
kontradikcije (označbe)
     ↓
povzetek
     ↓
rešitev

n8n Workflow: text -> embedding -> broswer use -> kontradikcije -> povzetek -> rešitev
