import pandas as pd
import requests
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

# Defining the function to call ChatGPT API
def call_chatgpt_api(prompt, api_key):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}]
    }
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            if 'choices' in response_json:
                return response_json['choices'][0]['message']['content']
            else:
                print("Unexpected response structure:", response_json)
                return None
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print("429 Too Many Requests: Retrying in 1 minute...")
                time.sleep(60)
            else:
                print("HTTPError:", e)
                return None
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            return None
    print("Failed to get a valid response after several attempts")
    return None

# Defining the function to retrieve additional information
def retrieve_additional_info(review):
    # List of positive and negative words
    positive_words = ["excellent", "amazing", "wonderful", "fantastic", "outstanding", "superb", "great", "good", "positive", "happy"]
    negative_words = ["poor", "terrible", "awful", "bad", "disappointing", "horrible", "negative", "sad", "angry", "worst"]

    # Adding positive and negative words to the additional information
    additional_info = f" Commonly positive words: {', '.join(positive_words)}. Commonly negative words: {', '.join(negative_words)}."
    return additional_info

# Defining the function to send a review to ChatGPT with RAG technique
def send_review_to_chatgpt_with_rag(review, api_key):
    additional_info = retrieve_additional_info(review)
    enriched_review = review + additional_info
    prompt = f"Analyze the following review and determine if it is positive or negative. Respond with 'positive' or 'negative':\n\nReview: {enriched_review}\n"
    response = call_chatgpt_api(prompt, api_key)
    if response:
        response = response.strip().lower()
        if 'positive' in response:
            return 1
        elif 'negative' in response:
            return 0
    return None

# Defining the function to display ASIN menu
def display_asin_menu(df):
    unique_asins = df['asin'].unique()
    asin_menu = {str(index + 1): asin for index, asin in enumerate(unique_asins)}
    for key, value in asin_menu.items():
        print(f"{key}: {value}")

    selected_asin = None
    while selected_asin is None:
        user_choice = input("Please select the number next to the ASIN of the product: ")
        selected_asin = asin_menu.get(user_choice, None)
        if selected_asin:
            return selected_asin
        else:
            print("Invalid selection. Please enter a valid choice.")

# Defining the function to filter reviews by ASIN
def filter_reviews_by_asin(df, asin):
    filtered_reviews = df[df['asin'] == asin].copy()
    filtered_reviews['reviewText'] = filtered_reviews['reviewText'].str.lower()
    return filtered_reviews

# Reading data from CSV file
df = pd.read_csv('Toys_and_Games_0.01.csv')

# Keeping only the 'asin', 'overall', and 'reviewText' columns, and removing rows with missing values
df = df[['asin', 'overall', 'reviewText']].dropna()

# Converting 'overall' ratings to 0 or 1
df['overall'] = df['overall'].apply(lambda x: 0 if x in [1, 2, 3] else 1)

# Displaying the ASIN selection menu
selected_asin = display_asin_menu(df)

# Filtering reviews based on the selected ASIN
filtered_reviews = filter_reviews_by_asin(df, selected_asin)
reviews_list = filtered_reviews['reviewText'].tolist()
y_true = filtered_reviews['overall'].tolist()

print(f"Found {len(reviews_list)} reviews for ASIN {selected_asin}.")

# Sending reviews to ChatGPT one by one using RAG
api_key = ''
y_pred = []

for review in reviews_list:
    sentiment = send_review_to_chatgpt_with_rag(review, api_key)
    if sentiment is not None:
        y_pred.append(sentiment)
    else:
        print(f"Failed to get sentiment for review: {review}")

if len(y_pred) == len(y_true):
    # Calculating metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Creating ROC and Precision-Recall curves
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred)
    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_pred)

    # Removing inf from thresholds
    thresholds_roc = thresholds_roc[np.isfinite(thresholds_roc)]

    # Plotting ROC curve
    plt.figure()
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    # Plotting Precision-Recall curve
    plt.figure()
    plt.plot(recall_curve, precision_curve, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    # Creating metrics table for additional thresholds
    additional_thresholds = np.linspace(0, 1, num=5)
    thresholds = sorted(set(thresholds_roc).union(set(thresholds_pr)).union(set(additional_thresholds)))
    metrics = []

    for threshold in thresholds:
        y_pred_threshold = [1 if prob >= threshold else 0 for prob in y_pred]
        accuracy = accuracy_score(y_true, y_pred_threshold)
        precision = precision_score(y_true, y_pred_threshold, zero_division=0)
        recall = recall_score(y_true, y_pred_threshold)
        f1 = f1_score(y_true, y_pred_threshold, zero_division=0)
        metrics.append((threshold, accuracy, precision, recall, f1))

    metrics_df = pd.DataFrame(metrics, columns=['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # Printing the results to the screen
    print(metrics_df)

    # Saving the metrics table to a CSV file
    metrics_df.to_csv('metrics_for_different_thresholds.csv', index=False)
    print("Metrics table saved as 'metrics_for_different_thresholds.csv'.")

else:
    print("The number of sentiments returned does not match the number of reviews.")
