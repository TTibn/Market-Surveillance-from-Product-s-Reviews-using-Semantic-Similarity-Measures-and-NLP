# Importing the necessary libraries
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample

# Adding the code for Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
sentiments = []

# Global list to store raw scores that need normalization
raw_scores = []

# Functions Definitions:

# Information content dataset
brown_ic = wordnet_ic.ic('ic-brown.dat')

def heatmap_color(val):
    return 'color: black' if val <= 5 else 'color: white'

# Removing synonyms and words with similar meaning
def reduce_synonyms(keywords):
    synonyms_set = set()
    reduced_keywords = []

    for keyword in keywords:
        synsets = wn.synsets(keyword)
        lemma_names = {lemma.name() for synset in synsets for lemma in synset.lemmas()}
        if not lemma_names & synonyms_set:
            reduced_keywords.append(keyword)
            synonyms_set.update(lemma_names)

    return reduced_keywords

from nltk.corpus import sentiwordnet as swn

def filter_keywords_with_sentiments(danger_keywords):
    swn_all_words = list(swn.all_senti_synsets())
    positive_negative_words = set()
    for senti_synset in swn_all_words:
        if senti_synset.pos_score() > 0.5 or senti_synset.neg_score() > 0.5:
            words = senti_synset.synset.lemma_names()
            for word in words:
                positive_negative_words.add(word.replace('_', ' ').lower())
    filtered_keywords = [word for word in danger_keywords if word in positive_negative_words]
    return filtered_keywords

def calculate_semantic_similarity(lemma1, lemma2, method):
    synsets_1 = wn.synsets(lemma1)
    synsets_2 = wn.synsets(lemma2)
    max_sim = -1.0

    similarity_methods = {
        "wup": lambda s1, s2: s1.wup_similarity(s2),
        "path": lambda s1, s2: s1.path_similarity(s2),
        "lch": lambda s1, s2: s1.lch_similarity(s2),
        "res": lambda s1, s2: s1.res_similarity(s2, brown_ic),
        "jcn": lambda s1, s2: s1.jcn_similarity(s2, brown_ic),
        "lin": lambda s1, s2: s1.lin_similarity(s2, brown_ic)
    }

    if method not in similarity_methods:
        print(f"Method {method} is not recognized.")
        return None

    for synset1 in synsets_1:
        for synset2 in synsets_2:
            try:
                semSim = similarity_methods[method](synset1, synset2)
                if semSim is not None and semSim > max_sim:
                    max_sim = semSim
            except Exception as e:
                continue

    if method in ["res", "jcn", "lch"]:
        if max_sim >= 0.0:
            raw_scores.append(max_sim)

    return max_sim

import os
from concurrent.futures import ThreadPoolExecutor

def calculate_similarity_parallel(review_tokens, danger_keywords, method):
    results_dict = {}
    num_cores = os.cpu_count() // 2
    print("The number of cores being used is:", num_cores)
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = {}
        for token in review_tokens:
            for keyword in danger_keywords:
                future = executor.submit(calculate_semantic_similarity, token, keyword, method)
                futures[future] = token
                
        for future, token in futures.items():
            result = future.result()
            if token not in results_dict or results_dict[token] < result:
                results_dict[token] = result

    return results_dict

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wn.ADJ, 'N': wn.NOUN, 'V': wn.VERB, 'R': wn.ADV, 'S': wn.ADJ}
    return tag_dict.get(tag, wn.NOUN)

def singularize_nouns(input):
    lemmatizer = WordNetLemmatizer()
    singular_lemmas = []
    for lemma in input:
        words = lemma.split()
        singular_words = [lemmatizer.lemmatize(word, pos='n') for word in words]
        singular_lemma = ' '.join(singular_words)
        singular_lemmas.append(singular_lemma)
    return singular_lemmas

def preprocess_review_token(review):
    lemmatizer = WordNetLemmatizer()
    review = review.replace("can't", 'can not').replace("won't", 'will not').replace("n't", ' not').replace("'m", ' am').replace("'s", ' is').replace("'re", ' are').replace("'ve", ' have').replace("'ll", ' will').replace("'d", ' would')
    words = word_tokenize(review.lower())
    words = [word for word in words if word.isalnum()]
    stopwords = ["the", "a", "an", "is", "be", "are", "and", "in", "on", "at", "to", "for", "from", "of"]
    words = [word for word in words if word not in stopwords]
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    return list(set(words))

def normalize_scores(scores):
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    if min_val == max_val:
        return [0.5 for _ in scores]
    return [(score - min_val) / (max_val - min_val) for score in scores]

def calculate_similarity_for_each_review_token(review_tokens, danger_keywords, method, brown_ic):
    similarities = []
    for review_token in review_tokens:
        preprocessed_review_token = preprocess_review_token(review_token)
        similarity_score = calculate_weighted_average_semantic_similarity(preprocessed_review_token, danger_keywords, method, brown_ic)
        similarities.append(similarity_score)
    return similarities

def calculate_weighted_average_semantic_similarity(tweet, danger_keywords, method, brown_ic):
    total_weight = 0.0
    total_sem_sim = 0.0
    lemmatizer = WordNetLemmatizer()
    processed_tweet_tokens = set([lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tweet])
    processed_danger_keywords = set([lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in danger_keywords])
    results = calculate_similarity_parallel(processed_tweet_tokens, processed_danger_keywords, method)
    raw_scores = list(results.values())
    normalized_scores = normalize_scores(raw_scores)
    for score in normalized_scores:
        weight = 0.0
        if score >= 0.9:
            weight = 0.1
        elif score >= 0.6:
            weight = 0.3
        elif score >= 0.4:
            weight = 0.5
        elif score >= 0.2:
            weight = 0.8
        else:
            weight = 1.0
        total_sem_sim += score * weight
        total_weight += weight
    return total_sem_sim / total_weight if total_weight != 0.0 else 0

def calculate_tfidf_cosine_similarity(lemma1, danger_keywords):
    texts = [lemma1] + danger_keywords
    vectorizer = TfidfVectorizer().fit(texts)
    lemma_vector = vectorizer.transform([lemma1])
    keywords_vector = vectorizer.transform(danger_keywords)
    cosine_similarities = cosine_similarity(lemma_vector, keywords_vector)
    max_similarity = cosine_similarities.max()
    return max_similarity

def calculate_precision_recall_fpr(y_true, y_scores, thresholds):
    precision_list, recall_list, fpr_list = [], [], []
    y_pred_list = []
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        y_pred_list.append(y_pred)
        tp = sum((p == 1) and (t == 1) for p, t in zip(y_pred, y_true))
        fp = sum((p == 1) and (t == 0) for p, t in zip(y_pred, y_true))
        fn = sum((p == 0) and (t == 1) for p, t in zip(y_pred, y_true))
        tn = sum((p == 0) and (t == 0) for p, t in zip(y_pred, y_true))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        fpr_list.append(fpr)
    return precision_list, recall_list, fpr_list

def calculate_metrics_for_thresholds(y_true, y_scores, thresholds):
    metrics = []
    best_metrics_df = pd.DataFrame(columns=['Method', 'Metric', 'Best Value', 'Threshold', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        metrics.append((threshold, accuracy, precision, recall, f1))
    metrics_df = pd.DataFrame(metrics, columns=['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    best_thresholds = {}
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        best_value = metrics_df[metric].max()
        best_threshold = metrics_df[metrics_df[metric] == best_value]['Threshold'].values[0]
        row = {
            'Metric': metric,
            'Best Value': best_value,
            'Threshold': best_threshold,
            'Accuracy': metrics_df[metrics_df[metric] == best_value]['Accuracy'].values[0],
            'Precision': metrics_df[metrics_df[metric] == best_value]['Precision'].values[0],
            'Recall': metrics_df[metrics_df[metric] == best_value]['Recall'].values[0],
            'F1 Score': metrics_df[metrics_df[metric] == best_value]['F1 Score'].values[0]
        }
        best_metrics_df = pd.concat([best_metrics_df, pd.DataFrame([row])], ignore_index=True)
    return metrics_df, best_metrics_df

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

def filter_reviews_by_asin(df, asin):
    filtered_reviews = df[df['asin'] == asin].copy()
    filtered_reviews['reviewText'] = filtered_reviews['reviewText'].str.lower()
    return filtered_reviews

# *****************   MAIN PROGRAM   *************************

df = pd.read_csv('Toys_and_Games_0.01.csv')

df1 = df

# Keeping only the 'overall' and 'reviewText' columns, and drop rows with missing values
df = df[['overall', 'reviewText']].dropna()

# Converting 'overall' ratings to 0 or 1
df['overall'] = df['overall'].apply(lambda x: 0 if x in [1, 2, 3] else 1)

# Balancing the dataset by oversampling
positive_reviews = df[df['overall'] == 1]
negative_reviews = df[df['overall'] == 0]
min_count = min(len(positive_reviews), len(negative_reviews))

balanced_positive_reviews = resample(positive_reviews, replace=True, n_samples=min_count, random_state=42)
balanced_negative_reviews = resample(negative_reviews, replace=True, n_samples=min_count, random_state=42)

balanced_df = pd.concat([balanced_positive_reviews, balanced_negative_reviews])

# Preprocessing the reviews to tokenize them
balanced_df['tokens'] = balanced_df['reviewText'].apply(preprocess_review_token)

# Joining the tokens back into strings to create the processed text
balanced_df['processed_text'] = balanced_df['tokens'].apply(lambda x: ' '.join(x))

# Initializing the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fitting the vectorizer on the processed text
tfidf_matrix = vectorizer.fit_transform(balanced_df['processed_text'])

# Printing the shape of the resulting matrix to confirm the number of features
print(tfidf_matrix.shape)

# Getting the feature names directly from the vectorizer
feature_names = vectorizer.get_feature_names_out()
print(feature_names)

# Filtering out non-English words using the NLTK words corpus
english_vocab = set(nltk.corpus.words.words())
danger_keywords = [word for word in feature_names if word in english_vocab]

print(danger_keywords)
print(len(danger_keywords))

danger_keywords_preprocessed = reduce_synonyms(danger_keywords)

print(danger_keywords_preprocessed)
print(len(danger_keywords_preprocessed))
print()

danger_keywords_preprocessed = filter_keywords_with_sentiments(danger_keywords_preprocessed)
print("Filtered Keywords Based on Sentiment:", danger_keywords_preprocessed)
print(len(danger_keywords_preprocessed))
print()

selected_asin = display_asin_menu(df1)

if selected_asin:
    filtered_reviews = filter_reviews_by_asin(df1, selected_asin)
    if not filtered_reviews.empty:
        print("Found reviews for the selected ASIN. Proceeding with the analysis...")
        
        print("Displaying reviews for ASIN:", selected_asin)
        
        for index, row in filtered_reviews.iterrows():
            print("\nReview:", row['reviewText'])
    else:
        print("No reviews found for the selected ASIN.")

print()
print("Printing the filtered_reviews", filtered_reviews)
print()
reviews_list = filtered_reviews['reviewText'].tolist()
print("Printing the reviews_list", reviews_list)

for review in reviews_list:
    vs = analyzer.polarity_scores(review)
    sentiment = 1 if vs['compound'] > 0 else 0
    sentiments.append(sentiment)
    print()
    print("Printing the sentiments", sentiments)

y_true = [1 if analyzer.polarity_scores(review)['compound'] > 0 else 0 for review in reviews_list]
print("Printing the y_true", y_true)

methods = ["wup", "res", "jcn", "lin", "path", "lch"]
thresholds = np.linspace(0, 1, 101)

# Store results for categorization
toy_categorization = {}

# DataFrame to store all precision, recall, and fpr data along with method and threshold
results_df = pd.DataFrame()
metrics_df_final = pd.DataFrame()
best_thresholds_df = pd.DataFrame()

# Repeat for each method and calculation of values for the Interpolated Precision-Recall and ROC Curves

for method in methods:
    y_scores = calculate_similarity_for_each_review_token(reviews_list, danger_keywords_preprocessed, method, brown_ic)
    print(f"Method:{method}, y_scores: {y_scores}")
    
    # Checking for diversity in y_true
    if len(set(y_true)) == 1:
        # Creating synthetic evaluations to enhance diversity
        fake_reviews_count = len(y_true)
        fake_sentiment = 0 if y_true[0] == 1 else 1
        fake_y_true = [fake_sentiment] * fake_reviews_count
        fake_y_scores = [0.5] * fake_reviews_count

        combined_y_true = y_true + fake_y_true
        combined_y_scores = y_scores + fake_y_scores

        precision_list, recall_list, fpr_list = calculate_precision_recall_fpr(combined_y_true, combined_y_scores, thresholds)
    else:
        precision_list, recall_list, fpr_list = calculate_precision_recall_fpr(y_true, y_scores, thresholds)
    
    avg_score = np.mean(y_scores)
    category = "Safe" if avg_score > 0.5 else "Dangerous"
    toy_categorization[method] = category
    print(f"The toy '{selected_asin}' is categorized as: {category} based on {method} method.")
    
    method_df = pd.DataFrame({
        'Threshold': thresholds,
        'Precision': precision_list,
        'Recall': recall_list,
        'FPR': fpr_list,
        'Method': method,
        'Category': category
    })
    
    results_df = pd.concat([results_df, method_df], ignore_index=True)
    
    recall_sorted_indices = np.argsort(recall_list)
    recall_sorted = np.array(recall_list)[recall_sorted_indices]
    precision_sorted = np.array(precision_list)[recall_sorted_indices]

    precision_interpolated = np.maximum.accumulate(precision_sorted[::-1])[::-1]

    plt.figure(figsize=(10, 5))
    plt.plot(recall_sorted, precision_interpolated, label=f'{method} Interpolated Precision-Recall', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Interpolated Precision-Recall Curve for {method}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f"Interpolated_Precision_Recall_Curve_for_{method}.jpg")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(fpr_list, recall_list, label=f'{method} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {method}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f"ROC_Curve_for{method}.jpg")
    plt.show()

    metrics_df, best_metrics = calculate_metrics_for_thresholds(y_true, y_scores, thresholds)
    metrics_df['Method'] = method
    metrics_df['Category'] = category

    best_metrics['Method'] = method
    best_thresholds_df = pd.concat([best_thresholds_df, best_metrics], ignore_index=True)

    # Append metrics dataframe to final metrics dataframe
    metrics_df_final = pd.concat([metrics_df_final, metrics_df], ignore_index=True)

# Save the final metrics results to a CSV file
csv_file_path_metrics = 'metrics_for_different_thresholds.csv'
metrics_df_final.to_csv(csv_file_path_metrics, index=False)

# Save the detailed results to another CSV file
csv_file_path_results = 'precision_recall_fpr_data_results.csv'
results_df.to_csv(csv_file_path_results, index=False)

# Save the best thresholds results to another CSV file
csv_file_path_best_thresholds = 'best_thresholds_results.csv'
best_thresholds_df.to_csv(csv_file_path_best_thresholds, index=False)

# Summary Comparative Charts

line_styles = ['-', '--', '-.', ':']
markers = ['o', '^', 's', 'p', '*', 'x', 'd']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

plt.figure(figsize=(12, 10))

for i, method in enumerate(methods):
    style = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]
    color = colors[i % len(colors)]

    recall_sorted_indices = np.argsort(results_df[results_df['Method'] == method]['Recall'])
    recall_sorted = np.array(results_df[results_df['Method'] == method]['Recall'])[recall_sorted_indices]
    precision_sorted = np.array(results_df[results_df['Method'] == method]['Precision'])[recall_sorted_indices]

    precision_interpolated = np.maximum.accumulate(precision_sorted[::-1])[::-1]

    plt.plot(recall_sorted, precision_interpolated, label=f'{method}', linestyle=style, color=color, marker=marker, markersize=5, linewidth=4)

plt.title('Interpolated Precision-Recall Curves', fontsize=20)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig("Interpolated_Precision_Recall_Curve.jpg")
plt.show()

plt.figure(figsize=(12, 10))
for i, method in enumerate(methods):
    style = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]
    color = colors[i % len(colors)]
    
    auc = np.trapz(sorted(results_df[results_df['Method'] == method]['Recall']), sorted(results_df[results_df['Method'] == method]['FPR']))

    plt.plot(sorted(results_df[results_df['Method'] == method]['FPR']), sorted(results_df[results_df['Method'] == method]['Recall']), label=f'{method} (AUC = {auc:.2f})', linestyle=style, color=color, marker=marker, markersize=5, linewidth=4)

chance_line, = plt.plot([0, 1], [0, 1], color='grey', linestyle='-.', linewidth=4, label='Chance')

plt.title('Comparative ROC Curves', fontsize=20)
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig("Comparative_ROC_Curves.jpg")
plt.show()

print("\nFinal Categorization Results:")
for method, category in toy_categorization.items():
    print(f"Method: {method}, Category: {category}")
