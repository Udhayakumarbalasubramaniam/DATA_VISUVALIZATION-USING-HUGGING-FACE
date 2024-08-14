import cohere
import pandas as pd

# Load Cohere API key from a secure location (file, environment variable, etc.)
with open('api.txt', 'r') as file:
    cohere_api_key = file.read().strip()

co = cohere.Client(cohere_api_key)

def get_llm_insights(df: pd.DataFrame) -> str:
    prompt = f"Provide a summary and insights for the following dataset:\n{df.head().to_dict()}"
    response = co.generate(
        model='command-xlarge-nightly',  # You can choose the appropriate model
        prompt=prompt,
        max_tokens=150
    )
    return response.generations[0].text.strip()

# Example usage with a DataFrame
df = pd.DataFrame({
    'Column1': [1, 2, 3, 4, 5],
    'Column2': ['A', 'B', 'C', 'D', 'E']
})

print(get_llm_insights(df))
