import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬",
    layout="wide"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 30px;
}

.movie-card {
    padding: 25px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.movie-title {
    font-size: 30px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Pydantic Model
# -----------------------------
class Movie(BaseModel):

    title: str

    release_year: Optional[int] = None

    genre: List[str]

    director: Optional[str] = None

    cast: List[str]

    rating: Optional[float] = None

    summary: str


# -----------------------------
# Output Parser
# -----------------------------
parser = PydanticOutputParser(
    pydantic_object=Movie
)


# -----------------------------
# Groq Model
# -----------------------------
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        Extract movie information from the paragraph.

        {format_instructions}
        """
    ),

    (
        "human",
        "{paragraph}"
    )
])


# -----------------------------
# UI
# -----------------------------

st.markdown(
    '<div class="title">🎬 Movie Information Extractor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Extract structured movie information using Groq + LangChain'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Input
# -----------------------------

st.subheader("📝 Enter Movie Paragraph")

paragraph = st.text_area(
    "Movie Description",

    placeholder=(
        "Example: Interstellar is a 2014 science fiction film "
        "directed by Christopher Nolan. It stars Matthew McConaughey, "
        "Anne Hathaway and Jessica Chastain..."
    ),

    height=180
)


# -----------------------------
# Button
# -----------------------------

if st.button(
    "🔍 Extract Movie Information",
    use_container_width=True
):

    if not paragraph.strip():

        st.warning(
            "⚠️ Please enter a movie paragraph first."
        )

    else:

        with st.spinner(
            "🤖 Extracting movie information..."
        ):

            try:

                # Create prompt
                final_prompt = prompt.invoke({

                    "paragraph": paragraph,

                    "format_instructions":
                        parser.get_format_instructions()

                })


                # Call Groq
                response = model.invoke(
                    final_prompt
                )


                # Parse response
                movie_data = parser.parse(
                    response.content
                )


                st.success(
                    "✅ Movie information extracted successfully!"
                )


                # -----------------------------
                # Movie Card
                # -----------------------------

                st.markdown(
                    '<div class="movie-card">',
                    unsafe_allow_html=True
                )


                st.markdown(
                    f'<div class="movie-title">'
                    f'🎬 {movie_data.title}'
                    f'</div>',
                    unsafe_allow_html=True
                )


                st.divider()


                # -----------------------------
                # Information Columns
                # -----------------------------

                col1, col2, col3 = st.columns(3)


                with col1:

                    st.markdown(
                        "### 📅 Release Year"
                    )

                    st.write(
                        movie_data.release_year
                        if movie_data.release_year
                        else "Not available"
                    )


                with col2:

                    st.markdown(
                        "### ⭐ Rating"
                    )

                    st.write(
                        movie_data.rating
                        if movie_data.rating
                        else "Not available"
                    )


                with col3:

                    st.markdown(
                        "### 🎭 Genre"
                    )

                    st.write(
                        ", ".join(movie_data.genre)
                        if movie_data.genre
                        else "Not available"
                    )


                st.divider()


                # -----------------------------
                # Director
                # -----------------------------

                st.markdown(
                    "### 🎥 Director"
                )

                st.write(
                    movie_data.director
                    if movie_data.director
                    else "Not available"
                )


                # -----------------------------
                # Cast
                # -----------------------------

                st.markdown(
                    "### 👥 Cast"
                )

                if movie_data.cast:

                    st.write(
                        ", ".join(movie_data.cast)
                    )

                else:

                    st.write(
                        "Not available"
                    )


                # -----------------------------
                # Summary
                # -----------------------------

                st.markdown(
                    "### 📖 Summary"
                )

                st.write(
                    movie_data.summary
                )


                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


                # -----------------------------
                # JSON
                # -----------------------------

                with st.expander(
                    "🔧 View Structured JSON"
                ):

                    st.json(
                        movie_data.model_dump()
                    )


            except Exception as e:

                st.error(
                    "❌ Something went wrong."
                )

                st.exception(e)