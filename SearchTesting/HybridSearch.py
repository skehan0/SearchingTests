import time
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from VectorSearch import read_document, tokenize_document

def keyword_search(keyword, file_path):
    start_time = time.time()  # Record start time
    try:
        with open(file_path, 'r') as file:
            line_number = 1
            for line in file:
                if keyword.lower() in line.lower():  # Case-insensitive search
                    end_time = time.time()  # Record end time
                    execution_time = end_time - start_time
                    memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)  # in megabytes

                    return {
                        "result": f"Found '{keyword}' in line {line_number}: {line.strip()}",
                        "approach": "Keyword Search",
                        "latency": execution_time,
                        "memory_usage": memory_usage,
                        "unit": "MB",
                    }

                line_number += 1
        end_time = time.time()  # Record end time if keyword is not found
        execution_time = end_time - start_time
        memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)  # in megabytes

        return {
            "result": f"Keyword '{keyword}' not found in the file.",
            "approach": "Keyword Search",
            "latency": execution_time,
            "memory_usage": memory_usage,
            "unit": "MB",
        }

    except FileNotFoundError:
        return {"result": f"File not found at path: {file_path}"}
    except Exception as e:
        return {"result": f"An error occurred: {str(e)}"}

def vector_search(query, sentences):
    start_time = time.time()
    try:
        vectorizer = TfidfVectorizer()
        query_vector = vectorizer.fit_transform([query])
        sentences_vectors = vectorizer.transform(sentences)
        similarity_scores = cosine_similarity(query_vector, sentences_vectors)
        end_time = time.time()
        execution_time = end_time - start_time
        memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)  # in megabytes

        return {
            "result": sentences[similarity_scores.argmax()],
            "approach": "Vector Search",
            "latency": execution_time,
            "memory_usage": memory_usage,
            "unit": "MB",
        }
    except Exception as e:
        return {"result": f"An error occurred: {str(e)}"}

# Example usage:
file_path = "document.txt"
document_text = read_document(file_path)
sentences = tokenize_document(document_text)

keyword_to_search = "retrieve relevant information"
result_keyword = keyword_search(keyword_to_search, file_path)
print(result_keyword)

query_to_search = "retrieve relevant information"
result_vector = vector_search(query_to_search, sentences)
print(result_vector)
