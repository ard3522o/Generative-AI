from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

from langchain_groq import ChatGroq

model = ChatGroq(model="openai/gpt-oss-120b")
prompt = ChatPromptTemplate.from_messages(
    [
    ("system",
     """You are an expert movie information assistant.

Step 1: Extract everything you can from the provided text.
Step 2: For any field missing from the text (cast, box office, budget, release year, awards, etc.),
use your own reliable knowledge of the movie to fill it in.

Rules:
1. Prefer the text over your own knowledge if they conflict.
2. Only fill a field from memory if you are confident. Otherwise return null or an empty list. Never invent numbers.
3. For box office and budget, give the figure with currency and mark it as approximate if unsure.
4. "cast" = real actors (with role). "characters" = fictional names.
5. List every field you filled from your own knowledge in "enriched_fields".
6. Return only the structured output.
7. Short Summary"""),
    ("human", 'Movie text:\n"""\n{text}\n"""\n\nExtract and complete the movie information.'),
]
)

para  = input("Give you paragraph")
final_prompt = prompt.invoke(
    {"text": para}
)
result = model.invoke(final_prompt)
print(result.content)