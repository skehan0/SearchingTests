import time
import psutil
from rake_nltk import Rake

def extract_text_from_file(file_path):
    try:
        # Open the file and read its content
        with open(file_path, 'r') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        return f"File not found at path: {file_path}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def keyword_extraction_rake(text):
    start_time = time.time()
    try:
        # Initialize Rake
        r = Rake()

        # Extract keywords from the text
        r.extract_keywords_from_text(text)

        # Get the ranked phrases with scores
        keywords_with_scores = r.get_ranked_phrases_with_scores()

        end_time = time.time()
        execution_time = end_time - start_time
        ram_usage = psutil.Process().memory_info().rss / 1024 / 1024  # RAM usage in MB

        return {
            "keywords_with_scores": keywords_with_scores,
            "execution_time": execution_time,
            "ram_usage": ram_usage
        }
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage:
file_path = "/Users/gavin.skehan/Documents/SearchTesting/document.txt"
text_to_extract = extract_text_from_file(file_path)
result_rake = keyword_extraction_rake(text_to_extract)

# Printing the keywords with scores
print("Keywords with scores: ")
for phrases, score in result_rake["keywords_with_scores"]:
    print(f"{phrases}: {score}")

print(f"Execution Time: {result_rake['execution_time']} seconds")
print(f"RAM Usage: {result_rake['ram_usage']} MB")
