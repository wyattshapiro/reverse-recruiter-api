import nltk
from nltk.corpus import stopwords
import pickle
import string
import re

CHOSEN_MODEL = 'RidgeClassifier-TfidfVectorizer'
CHOSEN_VECTORIZER_PATH = 'classifier/models/{}/{}'.format(CHOSEN_MODEL, 'vectorizer.pkl')
CHOSEN_CLASSIFIER_PATH = 'classifier/models/{}/{}'.format(CHOSEN_MODEL, 'classifier.pkl')


def preprocess_message(message):
    # initalize preprocess message variable
    preprocessed_message = message

    # remove names
    message_pos_tags = nltk.tag.pos_tag(preprocessed_message.split())
    message_no_pronouns = [word for word, tag in message_pos_tags if tag != 'NNP' and tag != 'NNPS']
    preprocessed_message = ' '.join(message_no_pronouns)

    # remove website urls
    label_unique_url = 'uniqueurl '
    preprocessed_message = re.sub(r'http\S+', label_unique_url, preprocessed_message)
    # all this pattern matching slows the code significantly
    # preprocessed_message = re.sub(r'www\S+', label_unique_url, preprocessed_message)
    # preprocessed_message = re.sub(r'\S+.*\.com', label_unique_url, preprocessed_message)
    # preprocessed_message = re.sub(r'\S+.*\.net', label_unique_url, preprocessed_message)
    # preprocessed_message = re.sub(r'\S+.*\.org', label_unique_url, preprocessed_message)

    # remove phone numbers
    label_unique_phone = 'uniquephone '
    preprocessed_message = re.sub(r'\S*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?', label_unique_phone, preprocessed_message)

    # remove punctuation
    message_nopunc = [char for char in preprocessed_message if char not in string.punctuation]
    preprocessed_message = ''.join(message_nopunc)

    # remove stopwords
    removed_words = stopwords.words('english')
    message_nopunc_nostopwords = [word for word in preprocessed_message.split() if word.lower() not in removed_words]
    preprocessed_message = ' '.join(message_nopunc_nostopwords)

    return preprocessed_message


def is_recruiter(message):
    # clean and vectorize this message
    message_clean = preprocess_message(message)
    message_vectorized = CHOSEN_VECTORIZER.transform([message_clean]).toarray()

    # predict if message is from a recruiter
    prediction = CHOSEN_CLASSIFIER.predict(message_vectorized)

    return prediction[0]


def download_model():
    # download stop words onto server
    nltk.download('stopwords')

    # read in model
    global CHOSEN_VECTORIZER 
    CHOSEN_VECTORIZER= pickle.load(open(CHOSEN_VECTORIZER_PATH, "rb"))
    global CHOSEN_CLASSIFIER 
    CHOSEN_CLASSIFIER = pickle.load(open(CHOSEN_CLASSIFIER_PATH, "rb"))


if __name__ == '__main__':
    pass
