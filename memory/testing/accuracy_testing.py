import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

try:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
except KeyError:
    print("ERROR: The 'OPENAI_API_KEY' environment variable is not set.")
    exit()


MODEL = "gpt-4o-mini"
DATA_FILE_PATH = "sp500_headlines_2008_2024.csv"
CONTEXT_TOKEN_LIMIT = 120000


def data(file_path):
    df = pd.read_csv(file_path)
    df = df[15000:]
    formatted = [
        f"News: {row['Title']} | Date: {row['Date']} | Closing Price: {row['CP']}" 
        for index, row in df.iterrows()
    ]

    return "\n".join(formatted)


def run_full_context(full_context):
    prompt = (
        "Based *only* on the following text, what is the news on the date 2023-12-07 and who is set to join the S&P 500? "
        "Do not use any other knowledge.\n\n"
        "--- START OF CONTEXT ---\n"
        f"{full_context}"
        "\n--- END OF CONTEXT ---"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    print("SUCCESS: The model processed the request.")
    print("Response:", response.choices[0].message.content)

if __name__ == "__main__":
    d = data(DATA_FILE_PATH)
    # print(d)
    run_full_context(d)
    