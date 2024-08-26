from sklearn.neighbors import NearestNeighbors
import numpy as np

from append_pool import load_pool, load_descriptions
from generate_tokens import get_embedding, load_model_and_tokenizer

def cosine_similarity(to_search, n_items):
    # Load the description pool and descriptions
    description_pool = load_pool()
    descriptions = load_descriptions()
    
    if not description_pool:
        raise ValueError("Description pool is empty. Add some descriptions first.")

    # Ensure n_items does not exceed the number of samples
    n_items = min(n_items, len(description_pool))

    # Load the model and tokenizer
    loaded_model, loaded_tokenizer = load_model_and_tokenizer()
    search_tokens = get_embedding(to_search, loaded_tokenizer, loaded_model)

    # Use k-NN to find closest descriptions
    knn = NearestNeighbors(n_neighbors=n_items, metric='cosine').fit(description_pool)
    distances, indices = knn.kneighbors([search_tokens])

    sorted_indices = indices[0][np.argsort(distances[0])]

    # Fetch the relevant descriptions based on indices
    relevant_descriptions = []
    for idx in sorted_indices:
        if idx < len(descriptions):  # Ensure index is within range
            relevant_descriptions.append(descriptions[idx])
        else:
            print(f"Index {idx} is out of range for descriptions.")

    return relevant_descriptions