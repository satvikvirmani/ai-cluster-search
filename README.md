# ⚖️ Legal Services Matcher

> An intelligent NLP-powered system that matches legal queries with the right legal specializations using semantic similarity.

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.44-yellow.svg)](https://huggingface.co/transformers/)

---

## 📋 Overview

Legal Services Matcher is a Flask-based web application that uses state-of-the-art Natural Language Processing to help users find the right legal specialist for their needs. By leveraging DistilBERT embeddings and cosine similarity, the system understands the semantic meaning of legal queries and matches them with relevant legal specializations.

### ✨ Key Features

- 🧠 **Semantic Understanding**: Uses DistilBERT for deep contextual understanding of legal queries
- 🔍 **Smart Matching**: k-Nearest Neighbors algorithm finds the most relevant legal specializations
- 💾 **Persistent Storage**: Automatically saves and loads embeddings for fast retrieval
- 🌐 **RESTful API**: Clean JSON API for easy integration
- 📊 **Pre-loaded Knowledge Base**: Comes with 29+ legal specializations out of the box

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pipenv (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd legal-services-matcher
   ```

2. **Install dependencies**
   ```bash
   pipenv install
   # or
   pip install -r requirements.txt
   ```

3. **Initialize the models**
   ```bash
   pipenv run python generate_tokens.py
   ```

4. **Load seed data (optional)**
   ```bash
   pipenv run python get_descriptions.py
   ```

5. **Start the server**
   ```bash
   pipenv run python app.py
   ```

The application will be available at `http://localhost:5000`

---

## 💻 Usage

### Web Interface

Visit `http://localhost:5000` in your browser to access the simple web interface.

### API Endpoints

#### 1. Add a Legal Description

```bash
POST /submit
Content-Type: application/json

{
  "description": "Handles complex business mergers and acquisitions"
}
```

**Response:**
```json
{
  "message": "Data received"
}
```

#### 2. Search for Matching Specialists

```bash
GET /api/search
Content-Type: application/json

{
  "to_search": "I need help with a car accident injury claim",
  "n_items": 3
}
```

**Response:**
```json
{
  "message": "Data received",
  "data": [
    "Handles cases involving physical or emotional harm caused by negligence or wrongful acts, seeking compensation for clients.",
    "Represents individuals accused of criminal offenses, working to protect their rights and seek a favorable outcome.",
    "Handles legal matters related to the healthcare industry, including medical malpractice, HIPAA compliance, and healthcare reform."
  ]
}
```

### Python Example

```python
import requests

# Search for a legal specialist
response = requests.get('http://localhost:5000/api/search', json={
    'to_search': 'divorce and child custody dispute',
    'n_items': 3
})

matches = response.json()['data']
for idx, match in enumerate(matches, 1):
    print(f"{idx}. {match}")
```

---

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         Flask API (app.py)          │
├─────────────────────────────────────┤
│  /submit  │  /api/search  │  /      │
└─────┬──────────────┬────────────────┘
      │              │
      ▼              ▼
┌─────────────┐  ┌──────────────────┐
│  append_    │  │   k_nearest.py   │
│  pool.py    │  │  (k-NN Search)   │
└──────┬──────┘  └────────┬─────────┘
       │                  │
       ▼                  ▼
┌──────────────────────────────────┐
│     generate_tokens.py           │
│   (DistilBERT Embeddings)        │
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│   Persistent Storage (PKL)       │
│  - description_pool.pkl          │
│  - description_list.pkl          │
│  - model.pkl                     │
│  - tokenizer.pkl                 │
└──────────────────────────────────┘
```

---

## 📁 Project Structure

```
legal-services-matcher/
│
├── app.py                    # Flask application & API routes
├── generate_tokens.py        # Model loading & embedding generation
├── append_pool.py           # Embedding storage management
├── k_nearest.py             # Similarity search implementation
├── get_descriptions.py      # Utility for loading seed data
│
├── descriptions.csv         # Seed data: legal specializations
│
├── templates/
│   ├── index.html          # Homepage
│   └── result.html         # Results page (unused)
│
├── static/
│   └── style.css           # Basic styling
│
├── Pipfile                 # Dependency management
├── Pipfile.lock
└── .gitignore
```

---

## 🔧 Technical Details

### Model Architecture

- **Base Model**: `distilbert-base-uncased`
- **Embedding Dimension**: 768
- **Pooling Strategy**: Mean pooling over last hidden states
- **Similarity Metric**: Cosine similarity

### Dependencies

| Package | Purpose |
|---------|---------|
| Flask | Web framework |
| Transformers | DistilBERT model |
| PyTorch | Deep learning backend |
| scikit-learn | k-NN algorithm |
| pandas | CSV data handling |

---

## 📊 Included Legal Specializations

The system comes pre-loaded with 29 legal specializations including:

- Divorce Lawyer
- Criminal Defense Lawyer
- Personal Injury Lawyer
- Real Estate Lawyer
- Corporate Lawyer
- Intellectual Property Lawyer
- Tax Lawyer
- Immigration Lawyer
- ... and 21 more!

---

## 🛠️ Development

### Running in Debug Mode

```bash
export FLASK_ENV=development
pipenv run python app.py
```

### Adding New Specializations

```python
from append_pool import tokenise_append_to_pool

tokenise_append_to_pool("Your new legal specialization description here")
```

---

## 🚧 Future Enhancements

- [ ] Replace pickle files with proper database (PostgreSQL/MongoDB)
- [ ] Add user authentication & authorization
- [ ] Implement caching for faster response times
- [ ] Create React-based frontend UI
- [ ] Add batch processing capabilities
- [ ] Implement API documentation (Swagger)
- [ ] Add comprehensive test suite
- [ ] Deploy to cloud (AWS/GCP/Azure)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Your Name**

- GitHub: [@satvikvirmani](https://github.com/satvikvirmani)
- LinkedIn: [satvikvirmani](https://linkedin.com/in/satvikvirmani)
- Email: virmanisatvik01@gmail.com

---

## 🙏 Acknowledgments

- HuggingFace for the excellent Transformers library
- The Flask team for the lightweight web framework
- DistilBERT authors for the efficient BERT variant

---

## 📞 Support

If you have any questions or run into issues, please open an issue on GitHub or contact the maintainer.

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ and ☕

</div>
