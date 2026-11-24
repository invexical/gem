import hmac
import os
from typing import Iterator

import streamlit as st
from google import genai
from google.genai import types


st.set_page_config(
    page_title="Gemini Chat",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 900px; padding-top: 2.2rem; padding-bottom: 3rem; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.18); }
      .hero { padding: .3rem 0 1.15rem 0; }
      .hero-kicker { color: #8b8b9b; font-size: .78rem; letter-spacing: .13em; text-transform: uppercase; }
      .hero-title { font-size: 2.35rem; font-weight: 750; letter-spacing: -.055em; line-height: 1.12; margin: .25rem 0; }
      .hero-subtitle { color: #92929e; font-size: .98rem; }
      div[data-testid="stChatMessage"] { border: 1px solid rgba(128,128,128,.14); border-radius: 16px; padding: .8rem 1rem; }
      div[data-testid="stChatInput"] { padding-bottom: .6rem; }
      .stButton button, .stDownloadButton button { border-radius: 10px; }
      .tiny-note { color: #8d8d98; font-size: .78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_secret(name: str, default: str = "") -> str:
    """Read a Streamlit secret first, then an environment variable."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or os.getenv(name, default))


# Optional lightweight password gate. Set APP_PASSWORD in Streamlit Secrets
# before deploying to a URL that other people might be able to visit.
app_password = read_secret("APP_PASSWORD")
if app_password:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("<div class='hero'><div class='hero-kicker'>Private workspace</div><div class='hero-title'>Gemini Chat</div><div class='hero-subtitle'>Enter your app password to continue.</div></div>", unsafe_allow_html=True)
        supplied_password = st.text_input("App password", type="password")
        if st.button("Unlock", type="primary", use_container_width=True):
            if hmac.compare_digest(supplied_password, app_password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("That password wasn't correct.")
        st.stop()


api_key = read_secret("GEMINI_API_KEY")
if not api_key:
    st.error("Gemini API key not found. Add GEMINI_API_KEY to Streamlit Secrets or your environment variables.")
    st.code('GEMINI_API_KEY = "your-api-key-here"', language="toml")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_client(key: str) -> genai.Client:
    return genai.Client(api_key=key)


client = get_client(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.markdown("## ✦ Gemini settings")
    model_id = st.text_input(
        "Model ID",
        value="gemini-3.8-flash",
        help="Change this to another Gemini model ID available to your API key.",
    ).strip()
    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Lower values are more consistent; higher values are more varied.",
    )
    system_instruction = st.text_area(
        "System instructions",
        value="You are a helpful, clear, and thoughtful assistant. Explain things step by step when useful.",
        height=130,
        help="Optional guidance that applies throughout this conversation.",
    )

    st.divider()
    if st.button("＋  New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        transcript = ["# Gemini Chat\n"]
        for message in st.session_state.messages:
            speaker = "You" if message["role"] == "user" else "Gemini"
            transcript.append(f"## {speaker}\n\n{message['content']}\n")
        st.download_button(
            "↓  Export conversation",
            data="\n".join(transcript),
            file_name="gemini-chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if app_password and st.button("Lock app", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    st.markdown("<div class='tiny-note'>Your API key is read on the server and is not placed in the page's JavaScript.</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Personal AI workspace</div>
      <div class="hero-title">Gemini Chat <span style="color:#8b82f6">✦</span></div>
      <div class="hero-subtitle">A simple chat interface powered by your Google AI Studio API key.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Render the conversation saved in this browser session.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def response_stream(history: list[dict], model: str, instructions: str, temp: float) -> Iterator[str]:
    """Stream a response using the full conversation as context."""
    contents = []
    for item in history:
        role = "model" if item["role"] == "assistant" else "user"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=item["content"])],
            )
        )

    config = types.GenerateContentConfig(
        system_instruction=instructions.strip() or None,
        temperature=temp,
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        chunk_text = getattr(chunk, "text", None)
        if chunk_text:
            yield chunk_text


prompt = st.chat_input("Message Gemini…")
if prompt:
    if not model_id:
        st.warning("Enter a model ID in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Gemini is thinking…"):
                answer = st.write_stream(
                    response_stream(
                        st.session_state.messages,
                        model_id,
                        system_instruction,
                        temperature,
                    )
                )
            if not answer:
                answer = "I didn't receive a text response. Try again or choose a different model."
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as exc:
            st.error(f"Gemini request failed: {exc}")
            st.caption("Check that your API key is valid and that the selected model is available to your key.")
