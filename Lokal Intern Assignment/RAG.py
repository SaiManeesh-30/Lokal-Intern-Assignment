import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging
import json

# Suppress transformers warnings for cleaner output
logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Function to load tickets from a file
def load_tickets_from_file(filepath="resolved_tickets.txt"):
    tickets = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                tickets.append(json.loads(line.strip()))
        print(f"Loaded {len(tickets)} tickets from {filepath}.")
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please create the file with resolved tickets.")
        tickets = []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from {filepath}: {e}. Please check file format.")
        tickets = []
    return tickets

# Load sample resolved support tickets from file
tickets = load_tickets_from_file()

# Exit if no tickets are loaded, as the bot won't function
if not tickets:
    print("No tickets loaded. Exiting.")
    exit()

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')


# Embed ticket problems
problem_texts = [ticket["problem"] for ticket in tickets]
embeddings = embedder.encode(problem_texts, convert_to_tensor=False)


# Build FAISS index
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))


# Load a more capable local LLM (distilgpt2)

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


# Ensure pad_token is set for distilgpt2 if it's not by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# Define the RAG bot function
def support_bot(user_query, top_k=2):
    # 1. Retrieval
    query_embedding = embedder.encode([user_query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    
    # Store retrieved resolutions along with their original problem for better matching
    retrieved_pairs = []
    for i in indices[0]:
        retrieved_pairs.append({
            "problem": tickets[i]["problem"],
            "resolution": tickets[i]["resolution"]
        })
    
    context_resolutions = [pair["resolution"] for pair in retrieved_pairs]
    context_text = "\n".join(context_resolutions)

    # 2. Augmentation & Generation
    prompt = f"""You are a concise support assistant.
Your task is to provide a solution to the user's issue.
Base your answer **strictly and only** on the provided past resolutions.
If a past resolution directly addresses the user's issue, provide that resolution.
If the resolutions do not contain enough information to directly answer the user's issue, state clearly that you cannot provide a specific resolution.
Do NOT generate any extra text, conversational filler, greetings, or apologies beyond the core answer.
Do NOT repeat the input prompt or user query in your response.

Past resolutions:
{context_text}

User issue: {user_query}
Resolution:""" 

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    max_gen_length = input_ids.shape[1] + 80 

    response = generator(
        prompt, 
        max_length=max_gen_length,
        temperature=0.4,  
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True, 
    )
    
    generated_text = response[0]["generated_text"]
    
    # 3. Post-processing: Extract only the part after "Resolution:"
    final_answer = "I apologize, but I cannot provide a specific resolution based on the available past tickets for this issue." 
    
    answer_prefix = "Resolution:"
    if answer_prefix in generated_text:
        answer_start_index = generated_text.find(answer_prefix) + len(answer_prefix)
        temp_answer = generated_text[answer_start_index:].strip()
        
        if "\n" in temp_answer:
            temp_answer = temp_answer.split("\n")[0].strip()
        
        if not temp_answer.lower().startswith("past resolutions:") and \
           not temp_answer.lower().startswith("user issue:") and \
           any(char.isalpha() for char in temp_answer): 
           
            final_answer = temp_answer
    else:
        temp_answer = generated_text.replace(prompt, "").strip()
        if "\n" in temp_answer:
            temp_answer = temp_answer.split("\n")[0].strip()
        
        if not temp_answer.lower().startswith("past resolutions:") and \
           not temp_answer.lower().startswith("user issue:") and \
           any(char.isalpha() for char in temp_answer):
            final_answer = temp_answer
            
    # Try to enforce direct answer if a clear match is found in retrieval
    # This is where the fix goes: encode before calculating similarity
    user_query_embedding = embedder.encode([user_query.lower()])[0] # Encode user query
    for pair in retrieved_pairs:
        problem_embedding = embedder.encode([pair["problem"].lower()])[0] # Encode retrieved problem
        # Now calculate similarity using the embeddings
        if np.dot(user_query_embedding, problem_embedding) / (np.linalg.norm(user_query_embedding) * np.linalg.norm(problem_embedding)) > 0.7: # Cosine similarity
            return pair["resolution"] # Return the exact resolution for high confidence matches
            
    if len(final_answer.split()) < 5 or not any(char.isalpha() for char in final_answer) or "cannot provide a specific resolution" in final_answer.lower():
        return "I am sorry, I cannot provide a specific resolution based on the available past tickets for this issue."

    if len(final_answer.split()) > 50: 
        sentences = final_answer.split('.')
        if len(sentences) > 1 and len(sentences[0]) > 20: 
            final_answer = sentences[0].strip() + "."
        else: 
            final_answer = " ".join(final_answer.split()[:50]).strip() + "..."

    return final_answer

# Interactive loop
print("Support Bot Ready! (Ask a question or type 'exit' to quit)")
while True:
    query = input("\nAsk your support question: ")
    if query.lower() == "exit":
        break
    
    answer = support_bot(query)
        
    print(f"\nBot: {answer}")
