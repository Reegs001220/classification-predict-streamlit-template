"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from sklearn.model_selection import train_test_split
import streamlit as st
import joblib,os
from pathlib import Path
import matplotlib as plt
import matplotlib.pyplot as pyplt

# Data dependencies
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
import time

# Vectorizer
# news_vectorizer = open("resources/tfidfvect.pkl","rb")
# tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
# tweet_cv = HashingVectorizer(ngram_range = (1,1))

# Load your raw data
raw = pd.read_csv("resources/downsampled_train.csv")
#function for importing markdown files into the streamlit app
def load_markdown_file(markdown_file):
	"""
	Reads the contents of a markdown file and returns the text as a string.

	:param markdown_file: A string representing the path to the markdown file.
	:type markdown_file: str
	:return: A string containing the contents of the markdown file.
	:rtype: str
	"""
	
	return Path(markdown_file).read_text()

# The main function where we will build the actual app
def main(df):

	downsampled_df = pd.read_csv("resources/downsampled_train.csv")

	"""Tweet Classifier App with Streamlit """

	# Download NLTK data (if not already downloaded)
	nltk.download('stopwords')
	nltk.download('punkt')
	nltk.download('wordnet')

	# Dictionary containing all emojis
	emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
			':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
			':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
			':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
			'@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
			'<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
			';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

	# Dictionary containing contractions
	contractions = {
		"ain't": "is not",
		"aren't": "are not",
		"can't": "cannot",
		"can't've": "cannot have",
		"'cause": "because",
		"could've": "could have",
		"couldn't": "could not",
		"couldn't've": "could not have",
		"didn't": "did not",
		"doesn't": "does not",
		"don't": "do not",
		"hadn't": "had not",
		"hadn't've": "had not have",
		"hasn't": "has not",
		"haven't": "have not",
		"he'd": "he would",
		"he'd've": "he would have",
		"he'll": "he will",
		"he'll've": "he will have",
		"he's": "he is",
		"how'd": "how did",
		"how'd'y": "how do you",
		"how'll": "how will",
		"how's": "how is",
		"I'd": "I would",
		"I'd've": "I would have",
		"I'll": "I will",
		"I'll've": "I will have",
		"I'm": "I am",
		"I've": "I have",
		"isn't": "is not",
		"it'd": "it would",
		"it'd've": "it would have",
		"it'll": "it will",
		"it'll've": "it will have",
		"it's": "it is",
		"let's": "let us",
		"ma'am": "madam",
		"mayn't": "may not",
		"might've": "might have",
		"mightn't": "might not",
		"must've": "must have",
		"mustn't": "must not",
		"needn't": "need not",
		"oughtn't": "ought not",
		"shan't": "shall not",
		"sha'n't": "shall not",
		"she'd": "she would",
		"she'd've": "she would have",
		"she'll": "she will",
		"she'll've": "she will have",
		"she's": "she is",
		"should've": "should have",
		"shouldn't": "should not",
		"so've": "so have",
		"so's": "so is",
		"that'd": "that would",
		"that'd've": "that would have",
		"that's": "that is",
		"there'd": "there would",
		"there'd've": "there would have",
		"there's": "there is",
		"they'd": "they would",
		"they'd've": "they would have",
		"they'll": "they will",
		"they'll've": "they will have",
		"they're": "they are",
		"they've": "they have",
		"wasn't": "was not",
		"we'd": "we would",
		"we'd've": "we would have",
		"we'll": "we will",
		"we'll've": "we will have",
		"we're": "we are",
		"we've": "we have",
		"weren't": "were not",
		"what'll": "what will",
		"what'll've": "what will have",
		"what're": "what are",
		"what's": "what is",
		"what've": "what have",
		"when's": "when is",
		"when've": "when have",
		"where'd": "where did",
		"where's": "where is",
		"where've": "where have",
		"who'll": "who will",
		"who'll've": "who will have",
		"who's": "who is",
		"who've": "who have",
		"why's": "why is",
		"why've": "why have",
		"will've": "will have",
		"won't": "will not",
		"won't've": "will not have",
		"would've": "would have",
		"wouldn't": "would not",
		"wouldn't've": "would not have",
		"y'all": "you all",
		"y'all'd": "you all would",
		"y'all'd've": "you all would have",
		"y'all're": "you all are",
		"y'all've": "you all have",
		"you'd": "you would",
		"you'd've": "you would have",
		"you'll": "you will",
		"you'll've": "you will have",
		"you're": "you are",
		"you've": "you have"
	}

	def preprocess_lemmatize(message):
		# Lowercase
		tweet = message.lower()

		# Replace all URLs with 'URL'
		tweet = re.sub(r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)", 'URL', tweet)

		# Replace all emojis
		for emoji in emojis.keys():
			tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])

		# Replace @USERNAME with 'USER'
		tweet = re.sub(r'@[^\s]+', 'USER', tweet)

		# Replace contractions
		tweet = ' '.join([contractions[word] if word in contractions else word for word in tweet.split()])

		#Remove consecutive duplicate characters
		tweet = re.sub(r"(.)\1+", r"\1", tweet)

		# Replace all non-alphanumeric characters
		tweet = re.sub(r"[^a-zA-Z0-9]", " ", tweet)

		# Tokenize and lemmatize
		tokens = word_tokenize(tweet)
		lemmatizer = WordNetLemmatizer()
		lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

		# Update the 'message' column in the original DataFrame
		row = ' '.join(lemmatized_words)

		return row



	

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	if selection == "Prediction":
		#give a list of models from the resources/pickles folder
		predictors = ["Logistic Regression", "SGD Classifier", "Support Vector Classifier" ]
		model = st.sidebar.selectbox("Choose A model", predictors)

		# Creates a main title and subheader on your page -
		# these are static across all pages
		st.title("Tweet Classifer")
		st.subheader(f"Climate change tweet classification using {model}")

	elif selection == "Information":
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification using Machine Learning")

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown(load_markdown_file("resources/info.md"))

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")

		#streamlit issue with if statements
		predictors = ["Logistic Regression", "SGD Classifier", "Support Vector Classifier" ]
		model = st.sidebar.selectbox("Choose A model", predictors)

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","")


		if st.button("Classify"):

			#downsample the data
			downsampled_df = pd.read_csv("resources/downsampled_train.csv")
			
			# Preprocess the user input
			process_text = preprocess_lemmatize(tweet_text)

			# Load the fitted vectorizer
			vectorizer = joblib.load(open(os.path.join("resources/pickles/vectorizer.pkl"), "rb"))

			# Transform the user input using the loaded vectorizer
			vect_text = vectorizer.transform([process_text]).toarray()

			st.empty()


			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			if model == "Logistic Regression":
				predictor = joblib.load(open(os.path.join("resources/pickles/lr.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			elif model == "SGD Classifier":
				predictor = joblib.load(open(os.path.join("resources/pickles/sgd.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			elif model == "Support Vector Classifier":
				predictor = joblib.load(open(os.path.join("resources/pickles/svm.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.

			sentiment = 0

			print("predictions", prediction)

			if prediction == -1:
				sentiment = "Anti"
			elif prediction == 0:
				sentiment = "Neutral"
			elif prediction == 1:
				sentiment = "Pro"
			elif prediction == 2:
				sentiment = "News"

			with st.spinner('Wait for it...'):
				time.sleep(1)
			st.success("Text Categorized as: {}".format(sentiment))


		dist_sentiment = pyplt.figure(figsize=(3,3), facecolor='none')
		color = ("yellowgreen", "red", "gold", "pink")
		wp = {'linewidth':2, 'edgecolor':"black"}
		class_counts = downsampled_df['sentiment'].value_counts()
		explode = (0.1,0.1,0.1,0.1)
		class_counts.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=color, startangle=90, wedgeprops=wp, explode = explode , label='')
		pyplt.title('Distribution Of Sentiments', fontsize=8)  # Change fontsize here

		# Change fontsize of labels
		pyplt.rc('font', size=8)  # controls default text sizes
		pyplt.rc('axes', titlesize=8)  # fontsize of the axes title
		pyplt.rc('axes', labelsize=8)  # fontsize of the x and y labels
		pyplt.rc('xtick', labelsize=8)  # fontsize of the tick labels
		pyplt.rc('ytick', labelsize=8)  # fontsize of the tick labels
		pyplt.rc('legend', fontsize=8)  # legend fontsize
		pyplt.rc('figure', titlesize=8)  # fontsize of the figure title

		with st.expander("**See Distribution of Sentiments**"):
			st.pyplot(dist_sentiment)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main(raw)
