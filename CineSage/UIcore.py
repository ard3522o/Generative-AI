from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="Movie Information Chatbot", page_icon="🎬", layout="centered")


class CastMember(BaseModel):
    actor: str = Field(description="Real actor name")
    character: str = Field(description="Character played")


class MovieInfo(BaseModel):
    title: Optional[str] = Field(None, description="Movie title")
    director: Optional[str] = Field(None, description="Director name")
    release_year: Optional[str] = Field(None, description="Release year")
    genre: Optional[List[str]] = Field(None, description="Genres, or null if unknown")
    summary: Optional[str] = Field(None, description="Short 2-3 sentence summary")
    cast: Optional[List[CastMember]] = Field(None, description="Main cast, actors with character names, or null if unknown")
    budget: Optional[str] = Field(None, description="Budget with currency, plain text like '$165 million (approx.)'")
    box_office: Optional[str] = Field(None, description="Worldwide box office with currency, plain text like '$677 million (approx.)'")
    awards: Optional[List[str]] = Field(
        None,
        description="Major awards as plain strings, e.g. 'Academy Award - Best Visual Effects (Won)', or null if unknown",
    )
    key_facts: Optional[List[str]] = Field(None, description="Important facts about the movie, or null if none")
    enriched_fields: Optional[List[str]] = Field(
        None,
        description="Fields not in the text that were filled from the model's own knowledge, or null if none",
    )


SYSTEM_PROMPT = """You are an expert movie information assistant.

Step 1: Extract everything you can from the provided text.
Step 2: For any field missing from the text (cast, box office, budget, release year, awards, etc.),
use your own reliable knowledge of the movie to fill it in.

Rules:
1. Prefer the text over your own knowledge if they conflict.
2. Only fill a field from memory if you are confident. Otherwise return null or an empty list. Never invent numbers.
3. For box office and budget, give the figure with currency and mark it as approximate if unsure.
4. Use plain text only in every field: no markdown, no backticks, no special formatting.
5. List every field you filled from your own knowledge in "enriched_fields".
6. Keep the summary short (2-3 sentences)."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", 'Movie text:\n"""\n{text}\n"""\n\nExtract and complete the movie information.'),
    ]
)


def render_movie(info: dict):
    st.subheader(info.get("title") or "Unknown movie")

    meta = []
    if info.get("director"):
        meta.append(f"🎥 Directed by **{info['director']}**")
    if info.get("genre"):
        meta.append("🎭 " + ", ".join(info["genre"]))
    if meta:
        st.markdown("  \n".join(meta))

    c1, c2, c3 = st.columns(3)
    c1.metric("Release year", info.get("release_year") or "N/A")
    c2.metric("Budget", info.get("budget") or "N/A")
    c3.metric("Box office", info.get("box_office") or "N/A")

    if info.get("summary"):
        st.markdown("**Summary**")
        st.write(info["summary"])

    if info.get("cast"):
        st.markdown("**Cast**")
        st.dataframe(
            [{"Actor": c["actor"], "Character": c["character"]} for c in info["cast"]],
            hide_index=True,
            use_container_width=True,
        )

    if info.get("awards"):
        st.markdown("**Awards**")
        for a in info["awards"]:
            st.markdown(f"- {a}")

    if info.get("key_facts"):
        st.markdown("**Key facts**")
        for f in info["key_facts"]:
            st.markdown(f"- {f}")

    if info.get("enriched_fields"):
        st.caption(
            "Filled from AI knowledge (not in your text): "
            + ", ".join(info["enriched_fields"])
            + ". Figures may be approximate."
        )


st.title("🎬 Movie Information Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "type": "text",
            "content": "Hi! Paste a paragraph about any movie and I'll extract the name, cast, genre, box office and more.",
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["type"] == "movie":
            render_movie(msg["content"])
        else:
            st.markdown(msg["content"])

user_input = st.chat_input("Paste your movie paragraph here...")

if user_input:
    st.session_state.messages.append({"role": "user", "type": "text", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
            chain = prompt | model.with_structured_output(MovieInfo)
            with st.spinner("Extracting..."):
                result = chain.invoke({"text": user_input})
            data = result.model_dump()
            render_movie(data)
            st.session_state.messages.append({"role": "assistant", "type": "movie", "content": data})
        except Exception as e:
            err = f"Something went wrong: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": err})