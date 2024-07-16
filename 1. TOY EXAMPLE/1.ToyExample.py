# LIBRARIES
import nltk
# nltk.download('all')
from collections import Counter
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pandas as pd
import re
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import textwrap
from textblob import TextBlob
from wordcloud import WordCloud
import string

# Extraction of main dangerous features contained in the tweets under examination
def extract_attributes(tweet):
    attributes = re.findall(r'\b[A-Za-z]+\b', tweet)
    return attributes

# Calculation of Wu-Palmer Semantic Similarity
def calculate_wup_similarity(word1, word2):
    # Retrieval of synsets for each word (lemma)
    word1_synsets = wn.synsets(word1)
    word2_synsets = wn.synsets(word2)

    max_sim = 0.0

    for synset1 in word1_synsets:
        for synset2 in word2_synsets:
            # Calculation of Wu-Palmer semantic similarity between synset1 and synset2
            sim = synset1.wup_similarity(synset2)
            if sim is not None and sim > max_sim:
                max_sim = sim

    # Check if at least one pair of synsets with semantic similarity was found
    if max_sim > 0.0:
        return max_sim
    else:
        return None

# Calculation of Part of Speech for a word via WordNet
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts."""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wn.ADJ,
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV
    }
    return tag_dict.get(tag, wn.NOUN)

# Conversion of Plural Nouns to Singular
def singularize_nouns(input):
    lemmatizer = WordNetLemmatizer()
    singular_lemmas = []

    for lemma in input:
        words = lemma.split()
        singular_words = [lemmatizer.lemmatize(word, pos='n') for word in words]
        singular_lemma = ' '.join(singular_words)
        singular_lemmas.append(singular_lemma)

    return singular_lemmas

## Tweet Preprocessing ##
def preprocess_tweet(tweet):

    lemmatizer = WordNetLemmatizer()

    # Expansion of word contractions
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

    # Splitting of Tweet into its constituting words
    words = word_tokenize(tweet)

    # Finding the tags of each tweet (if they exist)
    tags = re.findall(r'#\w+', tweet)

    for tag in tags:

        tag_words = tag.split('#')[1:] # Removal of the hashtag from the tags

        split_tags.extend(tag_words)
        # Splitting and storing of multiple Words from the tags (if they exist)
        words.extend(re.findall(r'[A-Z][a-z]*', tag))

    # Removal of multiple Tags from the word list
    words = ([word for word in words if word not in split_tags])

    # Conversion of Capitals to lowercase
    words = [word.lower() for word in words]

    # Removal of Punctuation
    words = [word for word in words if word.isalnum()]

    # Conversion of nouns to singular
    words = [lemmatizer.lemmatize(word, pos='n') for word in words]
    # Removal of stopwords
    stopwords = ["the", "a","A","an", "is", "be", "are", "and", "in", "on", "at", "to", "for", "from", "of"]
    tokens = [word for word in words if word not in stopwords]

    # Conversion of verbs to infinitive
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]

    # Removal of duplicate terms
    tokens = list(set(tokens))

    return tokens

# ## ** Calculation of Weighted Average SemSim ** ##
def calculate_weighted_average_semantic_similarity(tweet, danger_keywords):
    total_sem_sim = 0.0
    total_weight = 0.0

    lemmatizer = WordNetLemmatizer()
    lemmas = set()

    #** Separation of multi-word terms into primary nouns **
    for lemma in danger_keywords:
        lemma_words = lemma.split()
        lemmas.update(lemma_words)

    #** Determination of the part of speech of each dangerous term from WordNet and conversion to singular **
    danger_keywords = [lemmatizer.lemmatize(lemma, get_wordnet_pos(lemma)) for lemma in lemmas]

    # Removal of duplicate terms after separation
    danger_keywords = list(set(lemmas))

    # ** For each lemma in the tweet, SemSim is calculated with each danger term
    #/  and the maximum is stored between the lemma and the danger term
    #/  with the highest SemSim with this lemma
    for lemma in tweet:
        max_sem_sim = 0.0
        for danger_keyword in danger_keywords:
            sem_sim = calculate_wup_similarity(lemma, danger_keyword)
            if sem_sim is None:
                sem_sim = 0.001
            if sem_sim > max_sem_sim:
                max_sem_sim = sem_sim

        # ** WEIGHTS OF COEFFICIENTS **
        weight = 0.0
        if max_sem_sim >= 0.9:
            weight = 3.0
        elif max_sem_sim >= 0.7:
            weight = 2.0
        elif max_sem_sim >= 0.4:
            weight = 1.0
        elif max_sem_sim >= 0.2:
              weight = 0.5
        else:
            weight = 0.1

        total_sem_sim += max_sem_sim * weight
        total_weight += weight

    if total_weight == 0.0:
        return None

    #** Calculation of the Weighted Average for each Tweet **
    avg_sem_sim = total_sem_sim / total_weight
    return avg_sem_sim

# ** Function for Tweet Reconstruction after Preprocessing **
def reconstruct_tweet(tokens):
    reconstructed_tweet = ' '.join(tokens)
    return reconstructed_tweet

# ** Function to find related Tweets for the Toy the user is looking for **
def relevant_tweets_by_toy(toy_name):
    relevant_tweets = [tweet for tweet in example_tweets if toy_name.lower() in tweet.lower()]
    return relevant_tweets

#######################################
######## ** MAIN PROGRAM ** ########

# ** List of terms associated with danger according to EU legislation - Sample Terms **
danger_keywords = [
    'sodium', 'unsafe', 'drowning hazards', 'improperly labeled', 'suffocate',
    'product safety', 'explosion hazard', 'inadequate', 'inferior',
    'toxicity hazards', 'inaccurate', 'flammable', 'allergic reaction',
    'mutagenic substances', 'ingesting', 'age restrictions', 'safety assessment',
    'unsuitable age warnings', 'risky', 'asphyxiation hazard', 'pinch',
    'insubstantial', 'fire hazards', 'heat hazard', 'harmful substances',
    'radiation', 'radiological hazards', 'pinch points', 'radiation exposure',
    'dangerous', 'radiological', 'UV radiation', 'choking', 'suffocation',
    'inhalation', 'projectiles', 'fall hazards', 'hazard analysis', 'irritant',
    'ammonia', 'mechanical hazards', 'dye', 'ingestion', 'inadequate warnings',
    'cord hazards', 'safety guidelines', 'toxin', 'compliance monitoring',
    'button', 'labeling', 'inflammability hazards', 'piercing', 'toy manufacturing',
    'plastic', 'chemical hazards', 'consumer information', 'eye', 'noise hazards',
    'inadequate strength', 'physical', 'mutagenic', 'product testing',
    'developmental', 'sharp points', 'product recall',
    'cognitive development hazard', 'untested', 'faulty', 'risk assessment',
    'public awareness', 'declaration of conformity', 'ingest', 'cut',
    'mercury', 'endocrine disruptors', 'radiation hazards',
    'inhalation hazard', 'faulty design', 'responsible person',
    'strangulation', 'toy compliance', 'risk', 'genetic', 'poison',
    'cadmium', 'top-heavy', 'sharp edges', 'explosion', 'improper',
    'ingestion hazard', 'swallowing', 'burning', 'entanglement hazard',
    'danger', 'emotional', 'toxic', 'substandard', 'damaged',
    'quality assurance', 'hazardous', 'allergy', 'harmful',
    'strangulation hazard', 'defective', 'flammability', 'allergy hazards',
    'product documentation', 'consumer safety', 'asphyxiation hazards',
    'poor design', 'exposure to harmful substances',
    'ingestion of harmful substances', 'instability hazard',
    'risk evaluation', 'fire hazard', 'noise hazard',
    'thermal hazards', 'inflammation', 'unreliable', 'skin',
    'inadequately labeled', 'entanglement', 'choke',
    'child-resistant packaging', 'conformity assessment', 'cutting',
    'authorized representative', 'inflammability', 'electrocution',
    'chemical', 'drowning', 'pierce', 'radioactive materials', 'choking hazards',
    'arsenic', 'burn', 'product certification', 'electrical hazards',
    'explosion hazards', 'swallowing hazard', 'cutting hazards', 'asphyxiation',
    'physical development hazard', 'instructions for use', 'irritation',
    'CE marking', 'swallow', 'inhale', 'cord', 'strangulation hazards',
    'mandatory safety', 'carcinogenic', 'harmonized standards', 'laceration',
    'non-compliant', 'sensory', 'choke hazard', 'lead', 'chemical composition',
    'insufficient', 'noise', 'toy standards', 'unstable', 'aspiration',
    'electrocution hazard', 'child protection', 'explosive', 'excessive noise',
    'warning', 'bisphenol A', 'ergonomic hazards', 'EN standards',
    'genotoxic substances', 'falling', 'phthalates',
    'piercing hazards', 'toxic materials', 'fall', 'reproductive toxicity',
    'hazard', 'market surveillance', 'unsuitable', 'ingestion hazards',
    'safety evaluation', 'skin irritation', 'insufficient labeling',
    'enforcement measures', 'magnetic parts', 'toy industry', 'electric shock',
    'suffocation hazards', 'sharp', 'thermal', 'suffocation hazard',
    'hazard identification', 'quality control', 'biological hazards',
    'laceration hazard', 'hazards', 'toy safety regulation', 'electrical hazard',
    'small parts', 'eye injury', 'mandatory requirements', 'biological',
    'low-quality', 'age grading', 'safe play', 'regulatory compliance',
    'aspiration hazard', 'ergonomic', 'unapproved', 'emotional development hazard',
    'age appropriateness', 'carcinogenic substances', 'sensory development hazard',
    'formaldehyde', 'toxicity', 'product labeling', 'poisoning',
    'safety requirements', 'toy regulations', 'cognitive', 'unregulated']

# List of terms that are repeated and removed from the above list of dangerous terms
excluded_lemmas = ["toy", "development", "information","of","to","A"]

# Program section that separates terms with two or more lemmas from the above list
#// and then converts them to singular
all_keywords = []

for keyword in danger_keywords:
    # Word separation with spaces
    for word in keyword.split():
        # Check if the term is not similar to those in the excluded lemmas list
        if word not in excluded_lemmas:
            all_keywords.append(word)
danger_keywords_preprocessed = list(set(all_keywords))  # Removal of duplicate terms from the new list

# Conversion of processed terms of the new list to singular
danger_keywords_preprocessed = singularize_nouns(danger_keywords_preprocessed)

# Indicative Reviews with negative and positive feedback (Negative + Positive) - Synthetic Dataset
example_tweets = ["As a concerned parent, I appreciate the strict safety regulations that Toy1 adheres to. It gives me peace of mind knowing my child is playing with a safe toy. #Parenting #ToySafety",
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

similarity = calculate_wup_similarity("dangerous", "unsafe") #Παράδειγμα ελέγχου συναρτήσεων
print(similarity)

toy_attributes = []
reconstructed_tweets = []

# Section for toy name input
toy_name = input("Enter the name of a toy: ")

# Creation of a list for storing the toy name entered by the user
toy_list = []

# Conversion of the first letter to uppercase if the user types it in lowercase
if toy_name[0].islower():
    toy_name = toy_name.capitalize()

# Addition of the toy name to the list
toy_list.append(toy_name)
print(toy_list)

# Generation of relevant tweets for the specified toy
relevant_tweets = relevant_tweets_by_toy(toy_name)

# Storage of the number of relevant tweets
# Useful for subsequent visualizations
num_relev_tweets = len(relevant_tweets)

# Creation of a list tweets_number containing the respective number of tweets from the relevant ones found, e.g., ["1","2",..]
# Useful for subsequent visualizations
tweets_number = list(range(1, num_relev_tweets + 1))

# Visualization Diagrams

################**** 1st VISUALIZATION *****###################
# filtering of relevant tweets
if relevant_tweets:  # this checks if the list is not empty
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=[[tweet] for tweet in relevant_tweets],
             colLabels=[' Relevant Tweets'],
             cellLoc = 'center', loc='center', colColours=["#f5f5f5"])
    plt.savefig('relevant_tweets.jpg', format='jpg')
    plt.show()
else:
    print("No relevant tweets found for the given toy.")

###########***** Sentiment Analysis *****###################

sentiments = []
sentiment_polarities = []
negative_tweets = []  # List for storing negative tweets
positive_tweets = []  # List for storing positive tweets

for tweet in relevant_tweets:
    polarity = TextBlob(tweet).sentiment.polarity
    sentiment_polarities.append(polarity)
    if polarity >= 0:
        sentiments.append('positive')
        positive_tweets.append(tweet)
    elif polarity < 0:
        sentiments.append('negative')
        negative_tweets.append(tweet)  # Addition of negative tweet to the list

if negative_tweets:
    print("Negative tweets:")
    for tweet in negative_tweets:
        print(f"- {tweet}")
else:
    print("No negative tweets found.")

# Storage of the number of relevant negative tweets
# Useful for subsequent visualizations
num_tweets = len(negative_tweets)

# Creation of a list tweets_number containing the respective number of negative tweets from the relevant ones found
# Useful for subsequent visualizations
tweets_number = list(range(1, num_tweets + 1))

# Sentiment analysis execution and visualization
sentiments = ['positive' if TextBlob(tweet).sentiment.polarity >= 0 else 'negative' for tweet in relevant_tweets]
sentiments_df = pd.DataFrame(sentiments, columns=['Sentiment'])

################**** 2nd VISUALIZATION *****###################

plt.figure(figsize=(10,5))
sns.countplot(x='Sentiment', data=sentiments_df)
plt.title('Sentiment Analysis of Tweets for Toy', fontsize=26)
plt.xlabel('Sentiment', fontsize=24)
plt.ylabel('Count', fontsize=24)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('sentiment_analysis.jpg', format='jpg')
plt.show()

#########*******Calculation of SemSim values for NEGATIVE Tweets********** ###################
sem_sims = []
sem_sims_negative = []
i = 1
for tweet in negative_tweets:
    print("")
    print("******#############*********")
    print(f'******  {i}th Tweet   *********')
    print(tweet)
    print("")
    preprocessed_tweet = preprocess_tweet(tweet)
    print("The processed tweet under examination is:",preprocessed_tweet)
    print("")
    reconstructed_tweet = reconstruct_tweet(preprocessed_tweet)
    reconstructed_tweets.append(reconstructed_tweet)
    print("")
    avg_sem_sim = calculate_weighted_average_semantic_similarity(preprocessed_tweet, danger_keywords_preprocessed)
    sem_sims_negative.append(avg_sem_sim)
    sem_sims.append(avg_sem_sim)
    print(f"Tweet: {tweet}")
    i+=1
    if avg_sem_sim is not None:
        print()
        print(f"Weighted Average Semantic Similarity: {avg_sem_sim:.2f}")
        print()

# Average computation
avg_general = sum(sem_sims_negative) / num_relev_tweets

# Add categorization based on threshold
threshold = 0.5  # Example threshold value
category = "Dangerous" if avg_general > threshold else "Safe"
print(f"The product '{toy_name}' is categorized as: {category}")

################**** 3rd VISUALIZATION*****###################

for toy in toy_list:
    toy_tweets = [tweet for tweet, reconstructed_tweet in zip(negative_tweets, reconstructed_tweets) if toy.lower() in tweet.lower()]
    toy_attributes = []

    for tweet, reconstructed_tweet in zip(toy_tweets, reconstructed_tweets):
        preprocessed_tweet = preprocess_tweet(tweet)
        reconstructed_tweet = reconstruct_tweet(preprocessed_tweet)
        attributes = extract_attributes(reconstructed_tweet)
        relevant_attributes = [attr for attr in attributes if attr.lower() in danger_keywords_preprocessed]
        toy_attributes.extend(relevant_attributes)

    attribute_frequency = Counter(toy_attributes)

    plt.figure(figsize=(14, 10))

    # Random color assignment to the danger words
    colors = random.sample(list(cm.get_cmap('tab20b')(range(len(attribute_frequency)))), len(attribute_frequency))

    labels = []

    sorted_attributes = sorted(attribute_frequency.items(), key=lambda x: x[1], reverse=True)

    for i, (attribute, frequency) in enumerate(sorted_attributes):
        bar = plt.barh(i, frequency, color=colors[i % len(colors)])
        labels.append(f'{attribute}: {frequency}')

    plt.yticks(range(len(sorted_attributes)), [attribute for attribute, _ in sorted_attributes], fontsize=20)

    plt.xlabel('Frequency', fontsize=24)
    plt.title(f'Frequency of Dangerous Terms for {toy}', fontsize=26)
    plt.legend(labels, fontsize=18)

    plt.tight_layout()
    plt.savefig(f'frequency_of_dangerous_terms_{toy}.jpg', format='jpg')
    plt.show()

    print(f"\nNegative Tweets for {toy}:")
    for tweet in toy_tweets:
        print(tweet)
        print()

################**** 4th** VISUALIZATION*****###################
attribute_frequency = Counter(toy_attributes)

# Sort attributes by frequency in descending order
sorted_attributes = sorted(attribute_frequency.items(), key=lambda x: x[1], reverse=True)

# Plotting
plt.figure(figsize=(20, 12))
colors = plt.cm.get_cmap('tab20c', len(sorted_attributes))

# Creation of horizontal bar chart for each toy
for i, (attribute, frequency) in enumerate(sorted_attributes):
    color = colors(i % colors.N)
    plt.barh(attribute, frequency, color=color)

# Legend addition
legend_labels = [f'{attribute}: {frequency}' for attribute, frequency in sorted_attributes]
plt.legend(legend_labels, fontsize=18)

# Labels and title assignment
plt.xlabel('Frequency', fontsize=24)
plt.ylabel('Attribute', fontsize=24)
plt.title('Frequency of Danger-Related Words for Each Toy Attribute', fontsize=26)

# Plot display
plt.tight_layout()
plt.savefig('frequency_of_danger_related_words.jpg', format='jpg')
plt.show()


################**** 5th*** VISUALIZATION*****###################
attribute_frequency = Counter(toy_attributes)
fig, ax = plt.subplots(figsize=(19, 14))  # Diagram's size
colors = ['green', 'red', 'blue', 'orange', 'purple', 'yellow']

# Horizontal bars plotting
y_pos = range(len(attribute_frequency))
ax.barh(y_pos, attribute_frequency.values(), color=colors)

# Setting y-axis tick labels
ax.set_yticks(y_pos)
ax.set_yticklabels(attribute_frequency.keys(), fontsize=20)

# Explanation addition for bar colors
for i, freq in enumerate(attribute_frequency.values()):
    ax.text(freq, i, f'{freq}', va='center', fontsize=18)

# Title and labels setting
ax.set_title(f'Danger-related Words Frequency for {toy}', fontsize=26)
ax.set_xlabel('Frequency', fontsize=24)
ax.set_ylabel('Attribute', fontsize=24)

# Adjust layout to make room for the descriptions
fig.subplots_adjust(bottom=0.45)  # Increase bottom margin to make space for text

# Related tweets addition
separated_tweets = '\n'.join([tweet + '\n' + '-'*150 for tweet in toy_tweets])  # Add horizontal line after each tweet
fig.text(0.5, 0.04, separated_tweets, ha='center', fontsize=20, wrap=True)

plt.savefig(f'danger_related_words_frequency_{toy}.jpg', format='jpg')
plt.show()


#########*** Extraction and Visualization of Dangerous Terms via Wordcloud ***#################

# Conversion of tweets to lowercase and punctuation removal
tweets_lower = [tweet.lower().translate(str.maketrans('', '', string.punctuation)) for tweet in negative_tweets]

if toy_attributes:  # check if the list is not empty
    wordcloud = WordCloud().generate(' '.join(toy_attributes))
    plt.figure(figsize = (10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Dangerous Terms in Related Tweets for {toy}", fontsize=26)
    plt.axis("off")
    plt.savefig(f'dangerous_terms_wordcloud_{toy}.jpg', format='jpg')
    plt.show()
else:
    print("No dangerous terms found in the tweets.")


################**** 7th VISUALIZATION*****###################
plt.figure(figsize=(20, 14))  # Increase figure size
bars = plt.barh(tweets_number, sem_sims, color='maroon', align='center')

# Tweet text addition and formatting inside the bars
for i, bar in enumerate(bars):
    wrapped_text = textwrap.fill(negative_tweets[i], 100)  # wrap the text to fit inside the bar
    plt.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
             wrapped_text, ha='center', va='center', color='white', fontsize=18)

plt.xlabel("Semantic Similarity Value", fontsize=24)
plt.ylabel("Tweet Number", fontsize=24)
plt.title("Semantic Similarity Values for Each Tweet", fontsize=26)

# Upper right corner label addition
for i, v in enumerate(sem_sims):
    label = f"Tweet_{i+1} : {v:.2f}"
    plt.text(max(sem_sims) * 1.05, i + 1, label, color='black', va='center', fontsize=18)

plt.xlim(0, max(sem_sims) * 1.2)  # Extend x-axis for label accommodation

plt.savefig('semantic_similarity_values_each_tweet.jpg', format='jpg')
plt.show()



################**** 8th VISUALIZATION*****###################
fig, ax = plt.subplots(figsize=(20, 10))
bars = ax.bar(tweets_number, sem_sims, color='maroon', width=0.4)

# Labels addition on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2),
            ha='center', va='bottom', fontsize=18)

ax.set_xlabel("Tweet Number", fontsize=20)
ax.set_ylabel("Semantic Similarity Value", fontsize=20)
ax.set_title("Semantic Similarity Values for Each Tweet", fontsize=20)
ax.set_xticks(tweets_number)
ax.set_xticklabels([f"Tweet_{i}" for i in tweets_number], rotation=45, fontsize=20)

# Setting y-tick labels
plt.yticks(fontsize=20)

plt.tight_layout()
plt.savefig('semantic_similarity_values_each_tweet_bar.jpg', format='jpg')
plt.show()


