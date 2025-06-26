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
NUM_ROWS_TO_TEST = [50, 100, 200, 400, 800, 1200]
NUM_RUNS_PER_SIZE = 3 
DATA_FILE_PATH = "sp500_headlines_2008_2024.csv"



def load_and_prepare_data(file_path):
    try:
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        df_relevant = df[['Title', 'CP']].dropna()
        print(f"Successfully loaded {len(df_relevant)} rows with headlines and prices.")
        return df_relevant
    except FileNotFoundError:
        print(f"ERROR: The data file was not found at '{file_path}'.")
        return None
    except KeyError:
        print("ERROR: The CSV must contain 'Title' and 'CP' columns.")
        return None

def generate_text_from_data(dataframe, num_rows):
    if dataframe is None or len(dataframe) < num_rows:
        return ""


    data_chunk = dataframe.head(num_rows)
    
    formatted_lines = [
        f"News: {row['Title']} | Closing Price: {row['CP']}"
        for index, row in data_chunk.iterrows()
    ]
    

    return "\n".join(formatted_lines)


def run_latency_test(text_chunk):
    prompt = (
        "Based on the following news headlines and their corresponding S&P 500 closing prices, "
        "which news headline likely had the most significant impact on the market? "
        "Analyze the price changes between entries to support your answer.\n\n"
        f"{text_chunk}"
    )
    
    start_time = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100, 
        )
        end_time = time.monotonic()
        return {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "latency_sec": end_time - start_time,
        }
    except Exception as e:
        print(f"    An error occurred during API call: {e}")
        return None

def plot_results(results, model_name):
    if not results:
        print("\nNo results to plot.")
        return

    input_tokens = [r['input_tokens'] for r in results]
    latencies = [r['avg_latency_sec'] for r in results]
    throughputs = [r['throughput_tok_per_sec'] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle(f'LLM Performance Analysis: {model_name}', fontsize=16)

    ax1.plot(input_tokens, latencies, marker='o', linestyle='-', color='b')
    ax1.set_title('Latency vs. Input Size')
    ax1.set_xlabel('Number of Input Tokens')
    ax1.set_ylabel('Average Latency (seconds)')
    ax1.grid(True)

    ax2.plot(input_tokens, throughputs, marker='s', linestyle='-', color='r')
    ax2.set_title('Throughput vs. Input Size')
    ax2.set_xlabel('Number of Input Tokens')
    ax2.set_ylabel('Throughput (tokens/sec)')
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    print("\nDisplaying performance plots...")
    plt.show()


if __name__ == "__main__":
    market_data = load_and_prepare_data(DATA_FILE_PATH)
    
    if market_data is None:
        print("Exiting due to data loading failure.")
        exit()

    print("\n" + "-" * 60)
    print(f"Starting Latency Evaluation for model: {MODEL}")
    print(f"Testing number of rows: {NUM_ROWS_TO_TEST}")
    print(f"Runs per size: {NUM_RUNS_PER_SIZE}")
    print("-" * 60)
    
    all_results = []
    print(f"{'Num Rows':<15} | {'Actual Tokens':<15} | {'Avg Latency (s)':<20} | {'Throughput (tok/s)':<20}")
    print("-" * 60)

    for num_rows in NUM_ROWS_TO_TEST:
        text_chunk = generate_text_from_data(market_data, num_rows)
        if not text_chunk:
            print(f"Could not generate text for {num_rows} rows. Skipping.")
            continue
            
        print(f"Running tests for {num_rows} rows...")
        
        latencies_for_size = []
        actual_input_tokens = 0
        total_output_tokens = 0

        for i in range(NUM_RUNS_PER_SIZE):
            result = run_latency_test(text_chunk)
            if result:
                latencies_for_size.append(result["latency_sec"])
                actual_input_tokens = result["input_tokens"]
                total_output_tokens += result["output_tokens"]
        
        if not latencies_for_size:
            print(f"Could not get any results for size {num_rows}.")
            continue

        avg_latency = sum(latencies_for_size) / len(latencies_for_size)
        avg_output_tokens = total_output_tokens / len(latencies_for_size)
        throughput = (actual_input_tokens + avg_output_tokens) / avg_latency
        
        all_results.append({
            "input_tokens": actual_input_tokens,
            "avg_latency_sec": avg_latency,
            "throughput_tok_per_sec": throughput
        })

        print(f"{num_rows:<15} | {actual_input_tokens:<15} | {avg_latency:<20.4f} | {throughput:<20.2f}")

    print("-" * 60)
    print("Evaluation Complete.")
    
    plot_results(all_results, MODEL)