"""AI Data Analyst Agent - Streamlit app."""

import io

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.planner import get_api_key, plan_operation
from operations.cleaning import lowercase, remove_duplicates, strip_spaces
from operations.encoding import encode
from operations.missing import drop_missing, fill_missing
from operations.normalization import normalize
from utils.file_handler import read_file

load_dotenv()

st.set_page_config(page_title="AI Data Analyst Agent", page_icon="🤖", layout="wide")

UNKNOWN_MSG = (
    "❌ Sorry, I couldn't understand that operation.\n"
    "Try something like:\n"
    "- Remove missing values\n"
    "- Normalize Marks using z-score\n"
    "- Remove duplicates"
)
NO_FILE_MSG = "⚠️ Please upload a CSV or Excel file first."
API_ERROR_MSG = "❌ AI API error. Please check your API key or try again."

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
DEFAULTS = {
    "df": None,                 # current DataFrame
    "previous_df": None,        # only used for before / after comparison
    "show_compare": False,
    "chat_history": [],         # list of {"role": ..., "content": ...}
    "api_provider": "OpenAI",
    "api_key": "",              # key entered in the UI (never printed)
    "loaded_file": "",          # name+size of the file already in memory
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def active_api_key():
    """Key typed in the sidebar, otherwise fall back to the .env key."""
    if st.session_state.api_key:
        return st.session_state.api_key
    return get_api_key(st.session_state.api_provider)


# ---------------------------------------------------------------------------
# Operation executor - routes the JSON operation to a predefined function
# ---------------------------------------------------------------------------
def require_column(df, column):
    if not column or column not in df.columns:
        raise ValueError(f"Column '{column}' was not found in the dataset.")


def execute_operation(df, operation):
    """Run a predefined operation. Returns (new_dataframe, chat_message)."""
    kind = operation.get("operation")

    if kind == "drop_missing":
        before = int(df.isna().sum().sum())
        new_df = drop_missing(df.copy())
        after = int(new_df.isna().sum().sum())
        msg = (
            "✅ Done!\n\nRemoved rows with missing values.\n\n"
            f"Rows: {len(new_df)}\n"
            f"Missing values before: {before}\n"
            f"Missing values after: {after}"
        )
        return new_df, msg

    if kind == "fill_missing":
        column = operation.get("column")
        method = operation.get("method", "mean")
        require_column(df, column)
        before = int(df[column].isna().sum())
        new_df = fill_missing(df.copy(), column, method)
        after = int(new_df[column].isna().sum())
        msg = (
            "✅ Done!\n\n"
            f"Filled missing values in '{column}' using the {method}.\n\n"
            f"Rows: {len(new_df)}\n"
            f"Missing values before: {before}\n"
            f"Missing values after: {after}"
        )
        return new_df, msg

    if kind == "normalize":
        column = operation.get("column")
        method = operation.get("method", "zscore")
        new_df = normalize(df.copy(), column, method)
        label = "Z-score" if method == "zscore" else "Min-Max"
        msg = f"✅ Done!\n\nThe '{column}' column was normalized using {label} normalization."
        return new_df, msg

    if kind == "encode":
        column = operation.get("column")
        method = operation.get("method", "label")
        new_df = encode(df.copy(), column, method)
        if method == "one_hot":
            added = [c for c in new_df.columns if c.startswith(f"{column}_")]
            msg = f"✅ Done!\n\nOne-hot encoded '{column}'. New columns: {', '.join(added)}."
        else:
            msg = f"✅ Done!\n\nLabel encoded the '{column}' column."
        return new_df, msg

    if kind == "remove_duplicates":
        before = len(df)
        new_df = remove_duplicates(df.copy())
        msg = (
            "✅ Done!\n\nRemoved duplicate rows.\n\n"
            f"Rows before: {before}\n"
            f"Rows after: {len(new_df)}"
        )
        return new_df, msg

    if kind == "lowercase":
        column = operation.get("column")
        new_df = lowercase(df.copy(), column)
        return new_df, f"✅ Done!\n\nConverted '{column}' to lowercase."

    if kind == "strip_spaces":
        column = operation.get("column")
        new_df = strip_spaces(df.copy(), column)
        return new_df, f"✅ Done!\n\nRemoved extra spaces from '{column}'."

    if kind == "statistics":
        missing = int(df.isna().sum().sum())
        msg = (
            "Here are the dataset statistics:\n\n"
            f"- Total rows: {len(df)}\n"
            f"- Total columns: {len(df.columns)}\n"
            f"- Missing values: {missing}\n\n"
            "Full details are shown in the 📈 Dataset Statistics section."
        )
        return df, msg

    # unknown / unsupported
    return df, UNKNOWN_MSG


# ---------------------------------------------------------------------------
# Chat handling
# ---------------------------------------------------------------------------
def handle_prompt(prompt):
    """Process one chat message: plan with the LLM, then execute."""
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    if st.session_state.df is None:
        st.session_state.chat_history.append({"role": "assistant", "content": NO_FILE_MSG})
        return

    key = active_api_key()
    if not key:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "⚠️ Please add your API key in the sidebar first."}
        )
        return

    # 1) LLM planner: natural language -> JSON (never Python code)
    try:
        operation = plan_operation(
            st.session_state.api_provider,
            key,
            list(st.session_state.df.columns),
            prompt,
        )
    except ValueError:
        st.session_state.chat_history.append({"role": "assistant", "content": UNKNOWN_MSG})
        return
    except Exception:
        st.session_state.chat_history.append({"role": "assistant", "content": API_ERROR_MSG})
        return

    # 2) Python executor: JSON -> predefined pandas operation
    try:
        new_df, reply = execute_operation(st.session_state.df, operation)
    except ValueError as err:
        st.session_state.chat_history.append({"role": "assistant", "content": f"❌ {err}"})
        return
    except Exception:
        st.session_state.chat_history.append({"role": "assistant", "content": UNKNOWN_MSG})
        return

    # Keep only the current and previous DataFrame
    if new_df is not None and not new_df.equals(st.session_state.df):
        st.session_state.previous_df = st.session_state.df.copy()
        st.session_state.df = new_df
        st.session_state.show_compare = True

    st.session_state.chat_history.append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# Sidebar: AI settings + file upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ AI Settings")
    provider = st.radio("Select Provider", ["OpenAI", "Gemini"], key="api_provider")

    api_key_input = st.text_input("API Key", type="password", key="api_key_input")
    if st.button("Save API Key", width="stretch"):
        st.session_state.api_key = api_key_input
        if api_key_input:
            st.success(f"API key saved (****{api_key_input[-4:]})")
        else:
            st.info("API key cleared. Using .env key if available.")

    st.divider()

    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose CSV or Excel File", type=["csv", "xlsx", "xls"]
    )
    st.caption("Supported formats: .csv, .xlsx, .xls")

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}:{uploaded_file.size}"
        if file_id != st.session_state.loaded_file:
            try:
                st.session_state.df = read_file(uploaded_file)
                st.session_state.previous_df = None
                st.session_state.show_compare = False
                st.session_state.loaded_file = file_id
            except ValueError as err:
                st.error(f"❌ {err}")
            except Exception:
                st.error("❌ Could not read this file. Please upload a CSV or Excel file.")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🤖 AI Data Analyst Agent")
st.caption("Upload your dataset and describe what you want to do with your data.")

main_col, chat_col = st.columns([1.6, 1], gap="large")

df = st.session_state.df

# ---- Left column: preview + statistics + before/after ----
with main_col:
    if df is None:
        st.info("📂 Upload a CSV or Excel file in the sidebar to get started.")
    else:
        # Data preview
        st.markdown("#### 📊 Data Preview")
        info1, info2 = st.columns([4, 1])
        with info1:
            st.dataframe(df.head(10), width="stretch")
        with info2:
            st.metric("Rows", len(df))
            st.metric("Columns", len(df.columns))
        st.caption(
            "Rows: {}  |  Columns: {}".format(len(df), len(df.columns))
        )

        # Data types
        st.markdown("##### Column Types")
        dtypes = pd.DataFrame({"dtype": df.dtypes.astype(str)})
        st.dataframe(dtypes, width="stretch")

        st.divider()

        # Statistics
        st.markdown("#### 📈 Dataset Statistics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Rows", len(df))
        m2.metric("Total Columns", len(df.columns))
        m3.metric("Missing Values", int(df.isna().sum().sum()))

        numeric = df.select_dtypes("number")
        if not numeric.empty:
            st.markdown("**Statistics (Numeric Columns)**")
            st.dataframe(numeric.describe(), width="stretch")

        # Before / after comparison (only the last operation)
        if st.session_state.show_compare and st.session_state.previous_df is not None:
            st.divider()
            st.markdown("#### 🔄 Before / After")
            b1, b2 = st.columns(2)
            with b1:
                st.markdown("##### Before")
                st.dataframe(st.session_state.previous_df.head(10), width="stretch")
            with b2:
                st.markdown("##### After")
                st.dataframe(df.head(10), width="stretch")

# ---- Right column: chat + download ----
with chat_col:
    st.markdown("#### 💬 Chat")
    chat_box = st.container(height=430)
    with chat_box:
        if not st.session_state.chat_history:
            st.caption("Try: *Fill missing values in Marks with the mean.*")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if df is not None:
        st.markdown("#### ⬇️ Download Processed Data")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            data=csv_data,
            file_name="processed_data.csv",
            mime="text/csv",
            width="stretch",
        )
        # Excel download (optional, same data)
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        st.download_button(
            "📥 Download Excel",
            data=excel_buffer.getvalue(),
            file_name="processed_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

    prompt = st.chat_input("Ask anything about your dataset...")
    if prompt:
        handle_prompt(prompt)
        st.rerun()
