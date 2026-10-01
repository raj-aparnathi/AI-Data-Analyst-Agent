# 🤖 AI Data Analyst Agent

A small, beginner-friendly AI Data Analyst Agent built with **Python, Streamlit, and Pandas**.

Upload a CSV/Excel file, type what you want to do in plain English, and the app converts your instruction into a simple JSON operation, runs it with pandas, and shows the result.

## Features

- 📂 Upload CSV (`.csv`) and Excel (`.xlsx`, `.xls`) files
- 📊 Data preview, column types, and basic statistics
- 💬 ChatGPT-like chat input for natural-language instructions
- 🧠 LLM planner (OpenAI **or** Gemini) that returns structured JSON only
- 🐼 Pandas operations for missing values, normalization, encoding, and cleaning
- 🔄 Before / After comparison of the last operation
- ⬇️ Download the processed data as CSV or Excel
- 🔐 API keys from `.env` or entered in the sidebar (password field, never hardcoded)

## Project Structure

```text
AI-Data-Agent/
│
├── app.py                  # Streamlit UI + operation executor
│
├── agent/
│   └── planner.py          # Turns natural language into JSON operations
│
├── operations/
│   ├── missing.py          # drop / fill missing values
│   ├── normalization.py    # min-max, z-score
│   ├── encoding.py         # label, one-hot
│   └── cleaning.py         # duplicates, lowercase, strip spaces
│
├── utils/
│   └── file_handler.py     # Reads CSV / Excel uploads
│
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## API Key Setup

Copy the example file and add **one** key (you only need one provider):

```bash
cp .env.example .env
```

```text
OPENAI_API_KEY=sk-your-openai-key
GEMINI_API_KEY=your-gemini-key
```

You can also paste a key directly in the sidebar under **⚙️ AI Settings** — it is stored only for the current session and shown masked.

## How to Run

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Example Commands

Try instructions like:

```text
Fill missing values in Marks with the mean.
Remove rows with missing values.
Normalize the Age column using Z-score.
Normalize Marks using min-max.
Encode the Gender column using one hot encoding.
Remove duplicate rows.
Convert the Name column to lowercase.
Strip extra spaces from the Name column.
Show statistics of the dataset.
```

## How the Architecture Works

```text
                 ┌─────────────────┐
                 │      User       │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   Streamlit UI  │  Upload Dataset + Chat Input
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   LLM Planner   │  OpenAI / Gemini  ->  JSON only
                 └────────┬────────┘
                          │
                          ▼
                    Structured JSON
                          │
                          ▼
                 ┌─────────────────┐
                 │ Python/Pandas   │  predefined operation functions
                 │   Operations    │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Processed Data  │
                 └────────┬────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
           Preview     Statistics   Download
```

Key idea: **the LLM never writes or executes Python code.**

1. The user types a natural-language instruction.
2. `agent/planner.py` asks the LLM for a small JSON object, e.g.
   `{"operation": "fill_missing", "column": "Marks", "method": "mean"}`
3. `app.py` routes that JSON to a predefined pandas function.
4. The updated DataFrame is previewed, described, and can be downloaded.

## Supported Operations

| Operation | Example instruction |
|---|---|
| `fill_missing` | Fill missing Marks with the mean. |
| `drop_missing` | Remove rows with missing values. |
| `normalize` | Normalize Age using z-score. |
| `encode` | Encode Gender using one hot encoding. |
| `remove_duplicates` | Remove duplicate rows. |
| `lowercase` | Convert the Name column to lowercase. |
| `strip_spaces` | Strip extra spaces from Name. |
| `statistics` | Show statistics of the dataset. |
