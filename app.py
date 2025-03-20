from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize the FastAPI app
app = FastAPI()
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
@app.head("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}


# Define a request model
class ChatRequest(BaseModel):
    message: str

# Load your corpus JSON file (make sure it's in the same folder)
with open("teqnovation_chatbot_corpus.json", "r") as f:
    corpus_data = json.load(f)

# Extract questions and answers from the corpus
corpus_questions = []
corpus_answers = []
for entry in corpus_data:
    messages = entry.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            corpus_questions.append(msg.get("content"))
        elif msg.get("role") == "assistant":
            corpus_answers.append(msg.get("content"))

# Initialize the embedding model (free and lightweight)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings for the corpus questions
corpus_embeddings = embedder.encode(corpus_questions, convert_to_tensor=True)

def get_best_response(user_message: str, threshold: float = 0.5) -> str:
    # Compute embedding for the user's message
    user_embedding = embedder.encode([user_message], convert_to_tensor=True)
    # Calculate cosine similarity
    similarities = cosine_similarity(user_embedding.cpu().numpy(), corpus_embeddings.cpu().numpy())
    best_idx = np.argmax(similarities)
    best_similarity = similarities[0][best_idx]
    
    # Return the best matching answer if similarity is above threshold
    if best_similarity >= threshold:
        return corpus_answers[best_idx]
    else:
        return "Sorry, I didn't understand that. Can you please rephrase?"

@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    try:
        response_text = get_best_response(request.message)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
