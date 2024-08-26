from transformers import AutoTokenizer, AutoModel
import torch
import pickle

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')

# Load the tokenizer and model from pickle files
def load_model_and_tokenizer(model_filename='model.pkl', tokenizer_filename='tokenizer.pkl'):
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(tokenizer_filename, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    return model, tokenizer

# Tokenize and generate embeddings for each description
def get_embedding(text: str, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return torch.mean(last_hidden_states, dim=1).squeeze().detach().numpy()  # Mean pooling

if __name__ == '__main__':
    # Save the model and tokenizer
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('tokenizer.pkl', 'wb') as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)