from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

model = ChatGroq(model="openai/gpt-oss-120b")

result = model.invoke("Interstellar is a science-fiction movie directed by Christopher Nolan that explores space, time, love, and human survival. The story follows Cooper, a former NASA pilot who travels through a wormhole near Saturn with a team of astronauts to search for a new habitable planet because Earth is becoming difficult to live on. During their journey, they encounter strange planets, extreme time differences, and the mysteries of black holes. The movie beautifully shows the relationship between Cooper and his daughter Murph and highlights how love and human determination can connect people across time and space. With its stunning visuals, emotional story, and scientific concepts, Interstellar is considered one of the most thought-provoking science-fiction movies.Can you please extract the  summary and important facts about  the movie and it's  boxoffice collection")
print(result.text)