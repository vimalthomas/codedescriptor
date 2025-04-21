import os
import openai
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
OPENAI_API_KEY = ""  # replace or load from env
client = openai.OpenAI(api_key=OPENAI_API_KEY)

HTML_DOC_PATH = "./docs/final_report.html"
CHUNK_SIZE = 1500  # Approximate number of characters per chunk

# --- UTILITY FUNCTIONS ---

def load_and_clean_html(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    text = soup.get_text(separator='\n')
    return text

def chunk_text(text, chunk_size):
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def search_chunks(chunks, query):
    scores = []
    query = query.lower()

    for idx, chunk in enumerate(chunks):
        score = sum(query.count(word.lower()) for word in chunk.split())
        scores.append((score, idx))

    scores.sort(reverse=True)
    best_idx = scores[0][1]
    return chunks[best_idx]

def ask_llm(context, question):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering based on project documentation."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Failed to get response: {e}"

# --- MAIN INTERACTION ---

def main():
    print("Loading project documentation...")
    full_text = load_and_clean_html(HTML_DOC_PATH)
    chunks = chunk_text(full_text, CHUNK_SIZE)
    print(f"Documentation loaded with {len(chunks)} chunks.")

    while True:
        question = input("\nAsk a question about the project (or type 'exit'): ").strip()
        if question.lower() == "exit":
            break

        best_chunk = search_chunks(chunks, question)
        print("\nSearching project documentation...")

        answer = ask_llm(best_chunk, question)
        print("\nAnswer:")
        print("="*50)
        print(answer)
        print("="*50)

if __name__ == "__main__":
    main()
