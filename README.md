<!-- # aiXpravo
Ai x Pravo hekathon on FMF

Workflow:
* crawler (ip-rs)
* embedding generator (prednost je: hitrejše iskanje, šparanje na tokenih, semantična podobnost, manj halucinacij)
* vector database
* cosine similarity search (PCA graf)
* top 10 mnenj (z Broser Usom bomo iz 10 najbližjih extractal ključne podatke)
* LLM povzetek (Broswer Use) <=> Tax-Fin-Lex (keywords, info laws …)
* kontradikcije (označbe)
* povzetek
* rešitev -->

# ⚖️ AIxPravo — AI-powered Legal Intelligence

AIxPravo is an experimental legal AI system built during the **AI x Pravo Hackathon 2026** that enables fast, intelligent exploration of legal opinions issued by the **Information Commissioner of Slovenia**.

Instead of relying on traditional keyword search, AIxPravo understands **legal meaning** and retrieves the most relevant opinions using **semantic vector search, AI analysis, and automated reasoning**.

The system helps users answer complex legal questions by identifying the most relevant opinions, analyzing them, and highlighting possible contradictions between them.

---

# 🚀 The Vision

Legal knowledge is often hidden inside long documents, scattered across hundreds of opinions.

Finding the correct interpretation can require hours of reading.

**IP Asistant turns this process into seconds.**

Our system allows users to ask a legal question and instantly receive:

• the most relevant opinions  
• an explanation of their reasoning  
• insight into possible contradictions  
• contextual legal understanding  

---

# 🧠 Core Idea

Instead of asking AI to "know the law", we built a system that:

1. finds the **right legal sources**
2. analyzes them
3. explains them

The system combines **legal databases + semantic search + AI reasoning**.

---

# ⚙️ System Architecture

The system is built as a **multi-stage AI pipeline** designed to reduce hallucinations and maximize legal relevance.

User Question\
│\
▼\
Semantic Embedding\
│\
▼\
Vector Database Search\
│\
Top ~20 Most Similar Legal Opinions\
│\
▼\
Browser Use\
(collects context & extracts data)\
│\
▼\
Claude AI Reasoning

* relevance ranking

* contradiction detection

* synthesis\
│\
▼\
Final Structured Answer\
│\
▼\
Lovable Web App

---

# 📚 Our Legal Database

One of the strongest parts of the project is the **legal opinion database**.

We constructed a system capable of storing and searching through the full corpus of **Information Commissioner opinions**.

### Key advantages

• **Complete corpus of opinions**

• **Fast updates**
New legal opinions can easily be added without restructuring the system.

• **Vector-ready architecture**
Documents are embedded and indexed for semantic search.

• **AI-compatible structure**
The system is designed to feed documents efficiently into reasoning models.

This allows AI to operate on **real legal sources instead of generating answers from memory.**

---

# 🔎 Semantic Legal Search

Traditional legal search relies on **keywords**.

But legal questions rarely match the exact wording used in documents.

AIxPravo solves this using **vector embeddings**.

Each legal opinion is converted into a mathematical representation of its **meaning**.

When a user asks a question:

1. The question is embedded into a vector
2. Cosine similarity is computed
3. The system retrieves the **most semantically similar documents**

From hundreds of opinions, we immediately narrow the search space to roughly:

**~20 most relevant documents**

This dramatically improves both speed and accuracy.

---

# 🤖 AI Analysis Layer

Once the candidate documents are identified, AI performs deeper analysis.

### Browser Use

Browser Use:

• opens the retrieved legal opinions  
• extracts additional context  
• identifies relevant sections  

This ensures the reasoning model receives **complete context**.

---

### Claude Reasoning Model

Claude then analyzes all candidate documents and:

• ranks them by relevance  
• checks for **contradictions between opinions**  
• synthesizes the legal reasoning  
• generates a coherent explanation  

This step transforms raw legal texts into **interpretable legal insight**.

---

# 🌐 Web Interface

The final system is exposed through a simple web interface built with **Lovable**.

Users can:

• ask a legal question  
• instantly retrieve relevant opinions  
• explore explanations  
• inspect contradictions or agreements between sources

The interface focuses on **clarity, transparency, and source-based answers**.

---

# 🧩 Tech Stack

**AI / ML**

• Sentence Transformers (embeddings)  
• Claude (reasoning & synthesis)

**Retrieval**

• Vector database  
• Cosine similarity search

**Data Processing**

• Python  
• Document processing pipeline

**AI Agents**

• Browser Use

**Frontend**

• Lovable

---

# 🔬 Why This Approach Matters

Legal AI often struggles with hallucinations.

Our architecture avoids this by forcing the model to work with **real legal documents first**.

Key principles of our system:

• Retrieval before generation  
• Source grounding  
• Document comparison  
• Legal transparency

---

# 📈 Future Improvements

Possible next steps include:

• automated updates of new legal opinions  
• cross-referencing between different legal authorities  
• improved legal embeddings  
• multilingual support  
• advanced legal contradiction detection

---

# 👥 Team

Built during the **AI x Pravo Hackathon 2026**, an interdisciplinary event where teams combine legal and technical expertise to develop AI solutions for Odvetniška pisarna Ketler & Partnerji.

Our team combines:
* Lana Grudnik
* Jaka Furlan
* Tilen Goršek
* David Jagodič

---

# ⚡ Quick Example
**Question:**
Is a license plate considered personal data?

**IP Asistent** will:

1. search the legal opinion database

2. find the most relevant opinions

3. analyze them

4. detect agreement or contradictions

5. produce a structured explanation


---

# 🧠 Philosophy

AI should **assist legal reasoning**, not replace it.

AIxPravo helps users **navigate legal knowledge faster**, while keeping the **sources transparent and verifiable**.
