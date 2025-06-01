# 🧪 clinRAG

a langgraph-based RAG chatbot that performs semantic search over chunked clinical trial data (from [clinicaltrials.gov](https://clinicaltrials.gov))  

lightweight streamlit deployment @ [clinrag.streamlit.app](https://clinrag.streamlit.app)

---

## 🔄 flow

- 📄 parses and chunks (partial, for now) trial data from about ~1000 trials (i am on mongoDB free tier)  
- 🧠 embeds clinical trial chunks using `intfloat/e5-large-v2` (you can swap it out in `.env` if you're running locally)  
- 🗄️ stores data in mongoDB with a vector search index  
- 🔍 performs quick semantic search over the embedded chunks, with filtering based on:

  ```
  nctId
  status
  startDate (before/after)
  completionDate (before/after)
  studyType
  allocation
  interventionModel
  maskingType
  healthyVolunteers
  sex
  stdAges
  ```

- 🎛️ pretty streamlit ui for you to try out

---

### 🔬 overview of clinical trial parts:

1. **📘 OVERVIEW**  
 - general info: titles, description, status, dates, link to full study  
 - from `identificationModule`, `statusModule`, `descriptionModule`

2. **🧪 DESIGN**  
 - study type, phases, intervention model, allocation, masking, enrollment  
 - from `designModule`, `designInfo`, `maskingInfo`, `enrollmentInfo`

3. **🧍‍♂️ ELIGIBILITY**  
 - participant criteria: age range, sex, healthy volunteer status  
 - from `eligibilityModule`

4. **🧬 CONDITIONS**  
 - conditions studied + related keywords  
 - from `conditionsModule`

5. **🧫 ARMS & INTERVENTIONS**  
 - experimental/control groups and interventions (drugs, devices, etc)  
 - from `armsInterventionsModule`

6. **📊 OUTCOMES**  
 - primary/secondary outcomes: what’s being measured, when, and how  
 - from `outcomesModule` inside `protocolSection`

---

each of these parts is:
- 🧱 assembled into a text block  
- 🧠 embedded via `SentenceTransformer`  
- 📦 packaged into a `Chunk` with metadata
  
---

## 🧾 environment variables

create a `.env` file in the root directory like so:

```env
MONGODB_URI=<your cluster uri>
OPENAI_API_KEY=<your api key>
DATABASE_NAME=<self explanatory>
COLLECTION_NAME=<also self explanatory>
EMBEDDING_MODEL=<the sentence-transformers model you're using>
VECTOR_SEARCH_INDEX=<your mongoDB index name>
```

---

## 🛠️ setup

### 📋 requirements

- 🐍 python 3.12+  
- ⚡ [`uv`](https://github.com/astral-sh/uv) — a better pip  
- ☁️ a mongoDB cluster  

---

### 📦 installing dependencies

```bash
uv sync
```

---

## 🧹 preprocessing

1. **fetch + chunk**  
   🧺 pull and preprocess the trial data:

   ```bash
   uv run preprocessing/fetch_and_chunk.py
   ```

2. **initialize your db**  
   🧊 this inserts the chunks into mongoDB:

   ```bash
   uv run db_init.py
   ```

---

## 🧠 setting up vector search

you **must** create a vector search index in mongoDB atlas matching the schema in `extras/vector_index.json`  
⚠️ double-check that your `.env` variables (`MONGODB_URI`, `DATABASE_NAME`, etc) are correct

---

## 🚀 run the app

```bash
streamlit run app.py
```

---
