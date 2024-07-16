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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Adding the code for Sentiment Analysis 
analyzer = SentimentIntensityAnalyzer()
sentiments = []

# Global list to store raw scores that need normalization
raw_scores = []

# Information content dataset
brown_ic = wordnet_ic.ic('ic-brown.dat')

# Example dataset of tweets
example_tweets = [
    "As a concerned parent, I appreciate the strict safety regulations that Toy1 adheres to. It gives me peace of mind knowing my child is playing with a safe toy. #Parenting #ToySafety",
    "Attention parents! Toy2 has been recalled due to a potential choking hazard. Stay informed and keep your children safe. #ToyRecall #SafetyFirst",
    "Parents, beware of Toy2! Recent reports suggest it may contain lead paint, posing serious health risks. Let's demand safer options for our kids. #LeadFreeToys #ParentingAlert",
    "Beware of Toy3's excessive noise levels. Prolonged exposure can harm children's hearing. Let's prioritize their well-being and demand quieter toys. #NoiseHazard #ChildrensHealth",
    "Shocked to find out that Toy4 has small parts that pose a choking hazard. Let's hold toy companies accountable for ensuring child safety. #ChildProtection #ToyManufacturers",
    "Parents bad quality and high risk! Be careful of Toy1's easily breakable parts. They can become sharp objects, endangering our children. Choose toys built to last and prioritize safety. #BreakageRisk #ChildSafety",
    "Shocked to discover that Toy2 contains small parts that can easily be swallowed. This violates EU safety regulations. Protect our children! #ChokingRisk #ChildSafety",
    "Toy3's small magnets are a serious ingestion hazard. Manufacturers must ensure secure closures to prevent life-threatening accidents. #MagnetIngestion #ChildProtection",
    "Disappointed with the lack of durability in Toy1. It fell apart within days, and that's not what I expect from a toy marketed as safe. #QualityIssues #CustomerExperience",
    "Toy2's sharp edges are a serious concern. Children's toys should be designed with rounded edges to prevent injuries. #ChildSafety #DesignFlaws",
    "Toy2's packaging poses a suffocation risk. Let's advocate for child-safe packaging that eliminates hazards and prioritizes their well-being. #PackagingSafety #ChildProtection",
    "I regret purchasing Toy1. Its poor design and lack of safety features make it a potential danger to children. Spread the word to protect others! #UnsafeDesign #ToyFail",
    "Toy3 passed rigorous safety tests with flying colors. It's a relief to know our little ones can enjoy hours of fun without any worries. #ChildSafety #QualityToys",
    "Toy4's misleading age recommendation puts younger children at risk. Manufacturers should provide accurate guidelines to prevent accidents. #AgeAppropriateToys #ChildSafety",
    "Disappointed with Toy4's inadequate safety labeling. It's essential for parents to have clear information about potential hazards. Transparency is key! #LabelingIssues #ConsumerSafety",
    "Warning: Toy1 poses a strangulation risk due to long cords. Parents, please be cautious and ensure your children's safety. #ChokingHazard #UnsafeToys",
    "Just purchased Toy1 for my niece, and I'm relieved it meets all EU safety standards! #SafeToys #EURegulations",
    "Disappointed with Toy4's poor quality. It broke within minutes of playtime. Safety should be a top priority for all toy manufacturers. #UnsafeToys #CustomerReview",
    "Toy3 has passed all safety tests and even promotes educational development. It's a win-win for parents and children alike. #LearningThroughPlay #SafeToys",
    "It's alarming to hear that Toy4 contains small magnets that can be swallowed by children. Toy manufacturers, please prioritize safety over gimmicks. #ToyHazards #ChildProtection",
    "Shocking! Toy3's excessive small parts go against safety regulations. Children can easily choke on these hazards. Let's demand stricter enforcement. #ChokingRisk #ToySafetyStandards",
    "As a parent, I'm disturbed by Toy4's strong chemical odor. It raises concerns about toxic materials. Our children deserve safer, odor-free toys. #ChemicalSmell #ToxicToys",
    "Concerned about Toy2's lack of flame resistance. Fire safety should never be compromised when it comes to toys. Let's demand stricter regulations. #FireHazard #ToyFlammability",
    "Kudos to Toy1 for its eco-friendly design and non-toxic materials. It's essential to prioritize both safety and sustainability. #GreenToys #SafePlay",
    "Just discovered Toy2 contains harmful chemicals. Manufacturers need to be more transparent about the materials used in toys. #ChemicalSafety #ConsumerAwareness",
    "Parents, be wary of Toy4's sharp edges. They pose a serious risk of injury to our little ones. Demand safer toys for our children's well-being. #SafetyAlert #InjuryRisk",
    "Toy3 has received rave reviews for its high-quality construction and adherence to safety regulations. It's a must-have for any child's toy collection. #TopToy #CustomerFavorites",
    "My little one loves Toy3! It's not only entertaining but also meets all safety requirements. Thumbs up for a great toy! #HappyKids #SafetyStandards",
    "I'm appalled by the toxic chemicals found in Toy3. Manufacturers, prioritize our children's health and comply with safety standards! #ChemicalSafety #ChildrensHealth",
    "Bad news! Very concerned to learn that Toy1 has paint containing high levels of lead toxicity. This is a clear violation of toy safety regulations. Let's hold manufacturers accountable! #LeadToxicity #ChildHealth",
    "Toy2's flimsy construction raises concerns about its durability and safety. Parents, choose toys that withstand rigorous play and don't compromise on safety. #LowQuality #ToyDurability"
]

# Sentiment analysis
for tweet in example_tweets:
    vs = analyzer.polarity_scores(tweet)
    sentiment = 1 if vs['compound'] > 0 else 0
    sentiments.append(sentiment)

df = pd.DataFrame(list(zip(example_tweets, sentiments)), columns=['Tweet', 'Sentiment'])

# Balancing the dataset
positive_tweets = df[df['Sentiment'] == 1]
negative_tweets = df[df['Sentiment'] == 0]

min_count = min(len(positive_tweets), len(negative_tweets))

balanced_positive_tweets = positive_tweets.sample(n=min_count, random_state=42)
balanced_negative_tweets = negative_tweets.sample(n=min_count, random_state=42)

balanced_df = pd.concat([balanced_positive_tweets, balanced_negative_tweets])

csv_file_path = 'sentiment_analysis_tweets_balanced.csv'
balanced_df.to_csv(csv_file_path, index=False)

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

    if method == "cosine":
        return calculate_tfidf_cosine_similarity(lemma1, danger_keywords)

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
    """Map POS tag to first character lemmatize() accepts."""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wn.ADJ,
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'S': wn.ADJ
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

# Function to calculate the semantic similarity for each tweet
def calculate_similarity_for_each_tweet(tweets, danger_keywords, method, brown_ic):
    similarities = []
    for tweet in tweets:
        preprocessed_tweet = preprocess_tweet(tweet)
        similarity_score = calculate_weighted_average_semantic_similarity(preprocessed_tweet, danger_keywords, method, brown_ic)
        print(similarity_score)
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
            weight = 3.0
        elif score >= 0.6:
            weight = 2.0
        elif score >= 0.4:
            weight = 1.0
        elif score >= 0.2:
            weight = 0.5
        else:
            weight = 0.1

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
    print(f"y_pred_list: {y_pred_list}")
    print(f"Precision list: {precision_list}")
    print(f"Recall list: {recall_list}")
    print(f"FPR list: {fpr_list}")

    return precision_list, recall_list, fpr_list

# Function to calculate metrics for different thresholds
def calculate_metrics_for_thresholds(y_true, y_scores, thresholds):
    metrics = []
    best_thresholds = {
        'Accuracy': (0, 0),  # (best value, threshold)
        'Precision': (0, 0),
        'Recall': (0, 0),
        'F1 Score': (0, 0)
    }
    
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics.append((threshold, accuracy, precision, recall, f1))
        
        if accuracy > best_thresholds['Accuracy'][0]:
            best_thresholds['Accuracy'] = (accuracy, threshold)
        if precision > best_thresholds['Precision'][0]:
            best_thresholds['Precision'] = (precision, threshold)
        if recall > best_thresholds['Recall'][0]:
            best_thresholds['Recall'] = (recall, threshold)
        if f1 > best_thresholds['F1 Score'][0]:
            best_thresholds['F1 Score'] = (f1, threshold)
    
    return pd.DataFrame(metrics, columns=['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1 Score']), best_thresholds

def reconstruct_tweet(tokens):
    reconstructed_tweet = ' '.join(tokens)
    return reconstructed_tweet

def relevant_tweets_by_toy(toy_name):
    relevant_tweets = [tweet for tweet in example_tweets if toy_name.lower() in tweet.lower()]
    return relevant_tweets

# *****************   MAIN PROGRAM   *************************

danger_keywords = [
    'sodium', 'unsafe', 'alert', 'emergency', 'drowning hazards', 'improperly labeled', 'suffocate',
    'product safety', 'explosion hazard', 'inadequate', 'inferior',
    'toxicity hazards', 'unregulated'
]

excluded_lemmas = ["toy", "development", "information", "of", "to", "A"]

all_keywords = [word for keyword in danger_keywords for word in keyword.split() if word not in excluded_lemmas]

# Preprocess the danger_keywords
danger_keywords_preprocessed = singularize_nouns(list(set(danger_keywords)))
danger_keywords_preprocessed = reduce_synonyms(danger_keywords_preprocessed)
danger_keywords_preprocessed = filter_keywords_with_sentiments(danger_keywords_preprocessed)

print("Filtered Keywords Based on Sentiment:", danger_keywords_preprocessed)
print(len(danger_keywords_preprocessed))
danger_keywords_preprocessed = singularize_nouns(list(set(all_keywords)))

similarity = calculate_semantic_similarity("dangerous", "unsafe", "wup")
print(similarity)

# Inputting toy name and finding related tweets
toy_name = input("Enter the name of a toy: ").capitalize()
relevant_tweets = relevant_tweets_by_toy(toy_name)
y_true = [1 if analyzer.polarity_scores(tweet)['compound'] > 0.05 else 0 for tweet in relevant_tweets]
print("Printing the y_true", y_true)
print("Relevant tweets:")
print(relevant_tweets)

# Definition of methods and thresholds
methods = ["wup", "res", "jcn", "lin", "path", "lch"] # "cosine"]
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
best_thresholds_df = pd.DataFrame(columns=['Method', 'Metric', 'Threshold', 'Best Value', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Repeat for each method and calculation of values for the Interpolated Precision-Recall and ROC Curves
for method in methods:
    y_scores = calculate_similarity_for_each_tweet(relevant_tweets, danger_keywords_preprocessed, method, brown_ic)
    print(f"Method: {method}, y_scores: {y_scores}")
    precision_list, recall_list, fpr_list = calculate_precision_recall_fpr(y_true, y_scores, thresholds)
    
    # Save precision, recall, and fpr data
    precision_data[method] = precision_list
    recall_data[method] = recall_list
    fpr_data[method] = fpr_list

    # Categorization based on average score
    avg_score = np.mean(y_scores)
    category = "Dangerous" if avg_score > 0.5 else "Safe"
    toy_categorization[method] = category
    print(f"The toy '{toy_name}' is categorized as: {category} based on {method} method.")
    
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

    # Calculate metrics for different thresholds and get best thresholds
    metrics_df, best_thresholds = calculate_metrics_for_thresholds(y_true, y_scores, thresholds)
    metrics_df['Method'] = method
    metrics_df['Category'] = category

    # Append metrics dataframe to final metrics dataframe
    metrics_df_final = pd.concat([metrics_df_final, metrics_df], ignore_index=True)

    # Append best thresholds information to best_thresholds_df
    for metric, (best_value, threshold) in best_thresholds.items():
        best_thresholds_df = pd.concat([best_thresholds_df, pd.DataFrame({
            'Method': [method],
            'Metric': [metric],
            'Threshold': [threshold],
            'Best Value': [best_value],
            'Accuracy': [metrics_df.loc[metrics_df['Threshold'] == threshold, 'Accuracy'].values[0]],
            'Precision': [metrics_df.loc[metrics_df['Threshold'] == threshold, 'Precision'].values[0]],
            'Recall': [metrics_df.loc[metrics_df['Threshold'] == threshold, 'Recall'].values[0]],
            'F1 Score': [metrics_df.loc[metrics_df['Threshold'] == threshold, 'F1 Score'].values[0]]
        })], ignore_index=True)

# Save the best thresholds results to a CSV file
csv_file_path_best_thresholds = 'best_thresholds_for_each_metric.csv'
best_thresholds_df.to_csv(csv_file_path_best_thresholds, index=False)

# Save the final metrics results to a CSV file
csv_file_path_metrics = 'metrics_for_different_thresholds.csv'
metrics_df_final.to_csv(csv_file_path_metrics, index=False)

# Save the detailed results to another CSV file
csv_file_path_results = 'precision_recall_fpr_data_results.csv'
results_df.to_csv(csv_file_path_results, index=False)

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

    plt.plot(recall_sorted, precision_interpolated, label=f'{method}', linestyle=style, color=color, marker=marker, markersize=5, linewidth=2)

plt.title('Interpolated Precision-Recall Curves', fontsize=16)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig("Interpolated_PR_Curves_Advanced_SYNTHETIC.jpg")
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
plt.savefig("Comparative_ROC_Advanced_SYNTHETIC.jpg")
plt.show()

# Display final categorization results
print("\nFinal Categorization Results:")
for method, category in toy_categorization.items():
    print(f"Method: {method}, Category: {category}")
