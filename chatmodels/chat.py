from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
    temperature=0.8,
)
print("press 1 for funny AI agent ")
print("press 2 for a angry AI agent")
choice = int(input("Enter  the choice  required: "))
if choice == 1:
    mode = 'You are funny and happy ai  agent'
elif choice == 2:
    mode = "You are an angry AI agent and sad also"
messages = [
    SystemMessage(content=mode)
]

print("--------- Welcome! Type 0 to exit ---------")

while True:

    prompt = input("You: ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
        break

   

    result = model.invoke(messages)

    messages.append(AIMessage(content=result.text))

    print("Bot:", result.text)