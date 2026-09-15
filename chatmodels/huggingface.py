from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1"
)

model  = ChatHuggingFace(llm=llm)

result = model.invoke("who is Abhay Bhai?")
print(result.content)
