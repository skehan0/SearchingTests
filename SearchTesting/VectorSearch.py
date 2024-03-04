import time
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to read the document
def read_document(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        return f"File not found at path: {file_path}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to tokenize the document into sentences
def tokenize_document(text):
    sentences = text.split('. ')
    return sentences

# Function to perform vector search
def vector_search(query, sentences):
    start_time = time.time()
    try:
        vectorizer = TfidfVectorizer()
        query_vector = vectorizer.fit_transform([query])
        sentences_vectors = vectorizer.transform(sentences)
        similarity_scores = cosine_similarity(query_vector, sentences_vectors)
        end_time = time.time()
        execution_time = end_time - start_time
        ram_usage = psutil.Process().memory_info().rss / 1024 / 1024  # RAM usage in MB
        return {
            "similarity_scores": similarity_scores,
            "execution_time": execution_time,
            "ram_usage": ram_usage
        }
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage:
file_path = "document.txt"
document_text = read_document(file_path)
sentences = tokenize_document(document_text)

# Example query
query_to_search = "retrieve relevant information"
result = vector_search(query_to_search, sentences)

# Printing the similarity scores
for i, score in enumerate(result["similarity_scores"][0]):
    print(f"Sentence {i + 1}")
    print(f"Similarity score: {score}")

print(f"Execution Time: {result['execution_time']} seconds")
print(f"RAM Usage: {result['ram_usage']} MB")
