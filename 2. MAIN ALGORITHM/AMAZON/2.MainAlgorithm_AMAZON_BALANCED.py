# Libraries
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import re
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk


# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
sentiments = []

# Global list to store raw scores that need normalization
raw_scores = []

# Information content dataset
brown_ic = wordnet_ic.ic('ic-brown.dat')


# Preprocessing reviews
def preprocess_review(review):
    review = review.replace("can't", 'can not')
    review = review.replace("won't", 'will not')
    review = review.replace("n't", ' not')
    review = review.replace("'m", ' am')
    review = review.replace("'s", ' is')
    review = review.replace("'re", ' are')
    review = review.replace("'ve", ' have')
    review = review.replace("'ll", ' will')
    review = review.replace("'d", ' would')

    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(review.lower())
    words = [word for word in words if word.isalnum()]
    stopwords = set(nltk.corpus.words.words())
    words = [word for word in words if word not in stopwords]
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    return ' '.join(words)


# Function to calculate semantic similarity
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

    # if method == "cosine":
    #     return calculate_tfidf_cosine_similarity(lemma1, danger_keywords_preprocessed)

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

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wn.ADJ,
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV
    }
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

def preprocess_tweet(tweet):
    lemmatizer = WordNetLemmatizer()
    tweet = tweet.replace("can't", 'can not')
    tweet = tweet.replace("won't", 'will not')
    tweet = tweet.replace("n't", ' not')
    tweet = tweet.replace("'m", ' am')
    tweet = tweet.replace("'s", ' is')
    tweet = tweet.replace("'re", ' are')
    tweet = tweet.replace("'ve", ' have')
    tweet = tweet.replace("'ll", ' will')
    tweet = tweet.replace("'d", ' would')

    words = []
    split_tags = []
    words = word_tokenize(tweet)
    tags = re.findall(r'#\w+', tweet)

    for tag in tags:
        tag_words = tag.split('#')[1:]
        split_tags.extend(tag_words)
        words.extend(re.findall(r'[A-Z][a-z]*', tag))

    words = ([word for word in words if word not in split_tags])
    words = [word.lower() for word in words]
    words = [word for word in words if word.isalnum()]
    words = [lemmatizer.lemmatize(word, pos='n') for word in words]
    stopwords = ["the", "a", "A", "an", "is", "be", "are", "and", "in", "on", "at", "to", "for", "from", "of"]
    tokens = [word for word in words if word not in stopwords]
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    tokens = list(set(tokens))

    return tokens

def normalize_scores(scores):
    if not scores:
        return []

    min_val = min(scores)
    max_val = max(scores)
    if min_val == max_val:
        return [0.5 for _ in scores]

    return [(score - min_val) / (max_val - min_val) for score in scores]

def calculate_similarity_for_each_tweet(tweets, danger_keywords, method, brown_ic):
    similarities = []
    for tweet in tweets:
        preprocessed_tweet = preprocess_tweet(tweet)
        similarity_score = calculate_weighted_average_semantic_similarity(preprocessed_tweet, danger_keywords, method, brown_ic)
        similarities.append(similarity_score)
    return similarities

def calculate_weighted_average_semantic_similarity(tweet, danger_keywords, method, brown_ic):
    total_weight = 0.0
    total_sem_sim = 0.0

    lemmatizer = WordNetLemmatizer()
    lemmas = set()

    for lemma in danger_keywords:
        lemma_words = lemma.split()
        lemmas.update(lemma_words)

    danger_keywords = [lemmatizer.lemmatize(lemma, get_wordnet_pos(lemma)) for lemma in lemmas]
    danger_keywords = list(set(danger_keywords))

    raw_scores = []
    for lemma in tweet:
        max_sem_sim = 0.0
        for danger_keyword in danger_keywords:
            sem_sim = calculate_semantic_similarity(lemma, danger_keyword, method=method)
            if sem_sim is not None and sem_sim > max_sem_sim:
                max_sem_sim = sem_sim
        raw_scores.append(max_sem_sim)

    if method in ["res", "jcn", "lin", "lch"]:
        normalized_scores = normalize_scores(raw_scores)
    else:
        normalized_scores = raw_scores

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

    if total_weight != 0.0:
        return total_sem_sim / total_weight
    else:
        return 0

def calculate_tfidf_cosine_similarity(lemma1, danger_keywords):
    vectorizer = TfidfVectorizer().fit([lemma1] + danger_keywords)
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
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fpr = fp / (fp + tn) if fp + tn > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        fpr_list.append(fpr)
    return precision_list, recall_list, fpr_list

def calculate_metrics_for_thresholds(y_true, y_scores, thresholds):
    metrics = []
    best_thresholds = {}
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        metrics.append((threshold, accuracy, precision, recall, f1))

        # Saving the best values and corresponding thresholds
        if 'Accuracy' not in best_thresholds or accuracy > best_thresholds['Accuracy'][0]:
            best_thresholds['Accuracy'] = (accuracy, threshold, precision, recall, f1)
        if 'Precision' not in best_thresholds or precision > best_thresholds['Precision'][0]:
            best_thresholds['Precision'] = (precision, threshold, accuracy, recall, f1)
        if 'Recall' not in best_thresholds or recall > best_thresholds['Recall'][0]:
            best_thresholds['Recall'] = (recall, threshold, accuracy, precision, f1)
        if 'F1 Score' not in best_thresholds or f1 > best_thresholds['F1 Score'][0]:
            best_thresholds['F1 Score'] = (f1, threshold, accuracy, precision, recall)

    metrics_df = pd.DataFrame(metrics, columns=['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    return metrics_df, best_thresholds

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

# Main Program

# Reading data from CSV file
df = pd.read_csv('Toys_and_Games_0.01.csv')

# Keeping only the 'asin', 'overall', and 'reviewText' columns, and removing rows with missing values
df = df[['asin', 'overall', 'reviewText']].dropna()

# Converting 'overall' ratings to 0 or 1
df['overall'] = df['overall'].apply(lambda x: 0 if x in [1, 2, 3] else 1)

# Displaying the ASIN selection menu
selected_asin = display_asin_menu(df)

# Filtering reviews based on the selected ASIN
relevant_reviews = filter_reviews_by_asin(df, selected_asin)
print(relevant_reviews)
reviews_list = relevant_reviews['reviewText'].tolist()

# Balancing the dataset
positive_reviews = relevant_reviews[relevant_reviews['overall'] == 1]
negative_reviews = relevant_reviews[relevant_reviews['overall'] == 0]

min_count = min(len(positive_reviews), len(negative_reviews))

balanced_positive_reviews = resample(positive_reviews, replace=True, n_samples=min_count, random_state=42)
balanced_negative_reviews = resample(negative_reviews, replace=True, n_samples=min_count, random_state=42)

balanced_reviews = pd.concat([balanced_positive_reviews, balanced_negative_reviews])

print(f"Balanced dataset size: {len(balanced_reviews)}")

# Preprocessing the review text
balanced_reviews['processed_text'] = balanced_reviews['reviewText'].apply(preprocess_review)
y_true = balanced_reviews['overall'].tolist()

print(f"Found {len(balanced_reviews)} balanced reviews for ASIN {selected_asin}.")

# Initializing the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fitting the vectorizer on the processed text
tfidf_matrix = vectorizer.fit_transform(balanced_reviews['processed_text'])

# Getting the feature names directly from the vectorizer
feature_names = vectorizer.get_feature_names_out()

# Filtering out non-English words using the NLTK words corpus
english_vocab = set(nltk.corpus.words.words())
danger_keywords = [word for word in feature_names if word in english_vocab]

# Converting terms to singular
danger_keywords_preprocessed = singularize_nouns(list(set(danger_keywords)))

# Defining methods and thresholds
methods = ["wup", "res", "jcn", "lin", "path", "lch"]
thresholds = np.linspace(0, 1, 101)

# Store results for categorization
toy_categorization = {}

# Dictionaries to store precision, recall, and fpr data for saving to CSV
precision_data = {}
recall_data = {}
fpr_data = {}

# Dataframe to store all precision, recall, and fpr data along with method and threshold
results_df = pd.DataFrame()
metrics_df_final = pd.DataFrame()

# Dataframe to store best thresholds and corresponding metric values
best_thresholds_df = pd.DataFrame()

# Repeat for each method and calculation of values for the Interpolated Precision-Recall and ROC Curves
for method in methods:
    y_scores = calculate_similarity_for_each_tweet(balanced_reviews['processed_text'], danger_keywords_preprocessed, method, brown_ic)
    precision_list, recall_list, fpr_list = calculate_precision_recall_fpr(y_true, y_scores, thresholds)
    
    # Save precision, recall, and fpr data
    precision_data[method] = precision_list
    recall_data[method] = recall_list
    fpr_data[method] = fpr_list

    # Categorization based on average score
    avg_score = np.mean(y_scores)
    category = "Dangerous" if avg_score > 0.5 else "Safe"
    toy_categorization[method] = category
    # print(f"The toy with ASIN '{selected_asin}' is categorized as: {category} based on {method} method.")
    
    # Save results to dataframe
    method_df = pd.DataFrame({
        'Threshold': thresholds,
        'Precision': precision_list,
        'Recall': recall_list,
        'FPR': fpr_list,
        'Method': method,
        'Category': category
    })
    
    results_df = pd.concat([results_df, method_df], ignore_index=True)

    # Calculate metrics for all thresholds
    metrics_df, best_thresholds = calculate_metrics_for_thresholds(y_true, y_scores, thresholds)
    metrics_df['Method'] = method
    metrics_df['Category'] = category

    # Append metrics dataframe to final metrics dataframe
    metrics_df_final = pd.concat([metrics_df_final, metrics_df], ignore_index=True)

    # Store best thresholds and corresponding metric values in the dataframe
    for metric, values in best_thresholds.items():
        best_thresholds_df = pd.concat([best_thresholds_df, pd.DataFrame({
            'Method': [method],
            'Metric': [metric],
            'Best Value': [values[0]],
            'Threshold': [values[1]],
            'Accuracy': [values[2]],
            'Precision': [values[3]],
            'Recall': [values[4]],
            'F1 Score': [values[0]]
        })], ignore_index=True)

# Save the final metrics results to a CSV file
csv_file_path_metrics = 'metrics_for_different_thresholds.csv'
metrics_df_final.to_csv(csv_file_path_metrics, index=False)

# Save the detailed results to another CSV file
csv_file_path_results = 'precision_recall_fpr_data_results.csv'
results_df.to_csv(csv_file_path_results, index=False)

# Save the best thresholds and corresponding metric values to another CSV file
csv_file_path_best_thresholds = 'best_thresholds_results.csv'
best_thresholds_df.to_csv(csv_file_path_best_thresholds, index=False)

# Summary Comparative Charts

# Creation of lists for the style of lines and markers for each method
line_styles = ['-', '--', '-.', ':']
markers = ['o', '^', 's', 'p', '*', 'x', 'd']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Comparative Interpolated Precision-Recall Curve 
plt.figure(figsize=(8, 6))

for i, method in enumerate(methods):
    style = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]
    color = colors[i % len(colors)]

    recall_sorted_indices = np.argsort(recall_data[method])
    recall_sorted = np.array(recall_data[method])[recall_sorted_indices]
    precision_sorted = np.array(precision_data[method])[recall_sorted_indices]

    precision_interpolated = np.maximum.accumulate(precision_sorted[::-1])[::-1]

    # Add a starting point at (0,1)
    recall_sorted = np.insert(recall_sorted, 0, 0)
    precision_interpolated = np.insert(precision_interpolated, 0, 1)

    plt.plot(recall_sorted, precision_interpolated, label=f'{method}', linestyle=style, color=color, marker=marker, markersize=5, linewidth=2)

plt.title('Interpolated Precision-Recall Curves', fontsize=16)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig("Interpolated_Precision_Recall_Curve.jpg")
plt.show()

# Comparative ROC diagram
plt.figure(figsize=(8, 6))
for i, method in enumerate(methods):
    style = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]
    color = colors[i % len(colors)]
    
    auc = np.trapz(sorted(recall_data[method]), sorted(fpr_data[method]))

    plt.plot(sorted(fpr_data[method]), sorted(recall_data[method]), label=f'{method} (AUC = {auc:.2f})', linestyle=style, color=color, marker=marker, markersize=5, linewidth=2)

chance_line, = plt.plot([0, 1], [0, 1], color='grey', linestyle='-.', linewidth=2, label='Chance')

plt.title('Comparative ROC Curves', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig("Comparative_ROC_Curves.jpg")
plt.show()

# # Display final categorization results
# print("\nFinal Categorization Results:")
# for method, category in toy_categorization.items():
#     print(f"Method: {method}, Category: {category}")
