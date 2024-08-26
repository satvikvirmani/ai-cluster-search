import pickle
import os

from generate_tokens import load_model_and_tokenizer, get_embedding

# Define the filenames for storing the pool and descriptions
embedding_filename = 'description_pool.pkl'
description_filename = 'description_list.pkl'

def load_pool():
    """Load the pool from the pickle file. If the file does not exist, return an empty pool."""
    if os.path.exists(embedding_filename):
        with open(embedding_filename, 'rb') as file:
            return pickle.load(file)
    return []

def save_pool(embeddings):
    """Save the pool to the pickle file."""
    with open(embedding_filename, 'wb') as file:
        pickle.dump(embeddings, file)

def load_descriptions():
    """Load the descriptions from the pickle file. If the file does not exist, return an empty list."""
    if os.path.exists(description_filename):
        with open(description_filename, 'rb') as file:
            return pickle.load(file)
    return []

def save_descriptions(descriptions):
    """Save the descriptions to the pickle file."""
    with open(description_filename, 'wb') as file:
        pickle.dump(descriptions, file)

def tokenise_append_to_pool(text):
    # Load the model and tokenizer
    loaded_model, loaded_tokenizer = load_model_and_tokenizer()
    embedding = get_embedding(text, loaded_tokenizer, loaded_model)
    
    # Load the existing pool and descriptions
    embeddings = load_pool()
    descriptions = load_descriptions()

    # Append the new embedding and description
    embeddings.append(embedding)
    descriptions.append(text)

    # Save the updated pool and descriptions
    save_pool(embeddings)
    save_descriptions(descriptions)