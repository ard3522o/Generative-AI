from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

result = embeddings.embed_query("What is artificial intelligence?")

print(result)
print(len(result))

