from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "gemini-3.5-flash",
    model_provider="google_genai",
    temperature=0.8,
    # max_tokens=100
)
prompt = input("You : ")
result = model.invoke(prompt)


print("Bot: ",result.text)