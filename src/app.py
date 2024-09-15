import os
import json
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

# ANSI escape coes for colors
PINK = '\033[95m'
CYAN = '\033[65m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

KWOLEDGE_BASE = "./data/knowledge_base"
EMBEDDINGS_DB = "./data/embeddings_db"
CHUNK_STORE = "./data/chunks"

openai_key = "YOUR_OPENAI_KEY"


def initialize_openai(api_key):    
    return OpenAI(api_key=api_key)

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Check if the file is a PDF
        if filename.lower().endswith('.pdf'):
            try:
                # Open the file in binary mode
                with open(filepath, 'rb') as file:
                    reader = PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()  # Extract text from each page
                    documents.append(text)
            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")
        else:
            # Handle other file types (e.g., text files)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    documents.append(file.read())
                    
    return documents

def chunk_documents(documents, chunk_size=1000):
    chunks = []
    for doc in documents:
        start = 0
        while start < len(doc):
            chunk = doc[start:start + chunk_size].strip()
            chunks.append(chunk)
            start += chunk_size  # No overlap
    return chunks

def save_chunks(chunks, directory):
    os.makedirs(directory, exist_ok=True)
    for i, chunk in enumerate(chunks):
        chunk_file = os.path.join(directory, f"chunk_{i}.txt")
        with open(chunk_file, 'w', encoding='utf-8') as file:
            file.write(chunk)


# Function to generate embeddings using OpenAI's API
def generate_embeddings(client, chunks):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

def save_embeddings(embeddings, full_file_path):
    with open(full_file_path, 'w') as file:
        json.dump(embeddings, file)

def load_embeddings(directory, filename):
    with open(os.path.join(directory, filename), 'r') as file:
        embeddings = json.load(file)
    return embeddings

def find_relevant_chunk(query, client, chunks, embeddings):
    # Generate embedding for the query i.e. user input or questions
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Reshape query embedding to be 2D (1, N)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # Convert all chunk embeddings to a NumPy array
    chunk_embeddings = np.array(embeddings)

    # Calculate cosine similarity between query embedding and all chunk embeddings
    similarities = cosine_similarity(query_embedding, chunk_embeddings)

    # Find the index of the most similar chunk
    most_relevant_index = np.argmax(similarities)

    # Return the relevant chunk
    return chunks[most_relevant_index]

def generate_response(client, query, relevant_chunk):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Please be as accurate as possible and do not provide any guess answer. If you do not know the answer for sure, say so."},
            {"role": "user", "content": f"{relevant_chunk}\n\n{query}"}
        ]
    )
    return response.choices[0].message.content

def main():
    client = initialize_openai(openai_key)
    
    documents = load_documents(KWOLEDGE_BASE)
    print(f"{len(documents)} documents are loaded and ready to be chunked")
    
    chunks = chunk_documents(documents)
    print(f"Documents are successfully chunked into {len(chunks)} chunks")
    
    save_chunks(chunks, CHUNK_STORE)
    print(f"Chunks are saved to {CHUNK_STORE}")
    
    embeddings = generate_embeddings(client, chunks)
    
    # Save embeddings to a file inside the EMBEDDINGS_DB directory
    save_embeddings(embeddings, os.path.join(EMBEDDINGS_DB, "embeddings.json"))
    print(f"Embeddings saved to {EMBEDDINGS_DB}/embeddings.json")

    # what're the steps of loading the embeddings and perform the query from a user and generate response
    # 1. Load the embeddings from the file
    # embeddings = load_embeddings(EMBEDDINGS_DB, "embeddings.json")
    # print(f"Embeddings loaded from {EMBEDDINGS_DB}/embeddings.json")

    # 2. Get user input (query)
    while True:      

      query = input(YELLOW + "\nEnter your query (or 'q' to quit): " + RESET_COLOR)

      if query.lower() == 'q':
          print("Exiting...")
          break
    
      # 3: Get the relevant chunk based on the query
      relevant_chunk = find_relevant_chunk(query, client, chunks, embeddings)
      print(f"Relevant chunk size: {len(relevant_chunk)}\n\n")
      print(PINK + relevant_chunk + RESET_COLOR)

      # 4: Generate a response using the relevant chunk
      response = generate_response(client, query, relevant_chunk)
      
      # 5: Display the response to the user
      print("\nGenerated Response:\n")
      print(NEON_GREEN + f"\n{response}" + RESET_COLOR)
    
if __name__ == "__main__":
  main()