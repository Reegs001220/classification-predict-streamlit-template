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
import streamlit as st
import joblib,os
from pathlib import Path
import matplotlib as plt
import matplotlib.pyplot as pyplt

# Data dependencies
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
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
	options = ["Home", "Let's Talk Data...", "Let's Classify!", "Give Me Some Insights...", "About the Models", ]
	selection = st.sidebar.selectbox("Choose Option", options)

	if selection == "Home":
		# Cover Page
		st.title("SustainaMinds : Climate Change Tweet Classifier App")
		st.image("resources/imgs/Team_logo.jpg", use_column_width=True)  # Replace with the path to your image
		st.markdown(
    """
**Welcome to our Climate Change Tweet Classifier App!**

*SustainaMinds Data Collective: Who are we?*

- Reegan Rooke 
- Ayanda Moloi
- Kea Montshiwa
- Mohau Khanye
- Thabo Tladi
- Thabani Dhladhla

SustainaMinds Data Collective is a dynamic group committed to leveraging the power of data science for sustainable impact. With a focus on environmental consciousness,
our collective explores innovative solutions, conducts data-driven research, and builds tools like this Climate Change Tweet Classifier App. We strive to bridge the gap 
between data science and sustainability, fostering positive change and informed decision-making for a greener future.

SustainaMinds Data Collective has developed this app as a powerful tool for exploring sentiments behind climate-related tweets.
By leveraging our innovative platform and selecting from three top models, businesses can decipher whether a tweet supports, opposes, or falls in between the belief that 
climate change is man-made and real.

Unlock valuable insights into diverse perspectives shared on social media about the urgent matter of climate change.
Businesses can use these insights to adapt and refine their marketing strategies, ensuring alignment with public sentiments and contributing to a more sustainable future.

Join us in understanding and utilizing the wealth of information available through social media to make informed decisions and drive positive change!

**Some things to know about the app:**

- The app is divided into 5 sections: Home, Let's Talk Data..., Let's Classify!, Give Me Some Insights..., and About the Models.

- The **Home** page is a brief introduction to the app and the SustainaMinds Data Collective.
- The **Let's Talk Data...** page provides a brief overview of the data used to train the models as well as what each sentiment class represents. You can also select to see the raw data used.
- The **Let's Classify!** page is where you can use the app to classify your own tweets.
- The **Give Me Some Insights...** page provides some insights into the data that was discovered during the exploratory data analysis phase of the project.
- The **About the Models** page provides a brief overview of the models used in the app.

These sections can be accessed with the drop-down sidebar on the left of the app.
    """
)


	elif selection == "Let's Talk Data...":
		st.title("What Are We Working With?")
		st.image("resources/imgs/notebook_head_image.jpg", use_column_width=True)

		# Building out the "Information" page
	if selection == "Let's Talk Data...":
		# You can read a markdown file from supporting resources folder
		st.markdown(load_markdown_file("resources/info.md"))

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page


	elif selection == "Let's Classify!":
		#give a list of models from the resources/pickles folder
		predictors = ["Logistic Regression", "SGD Classifier", "Support Vector Classifier" ]
		model = st.sidebar.selectbox("Choose A model", predictors)

		# Creates a main title and subheader on your page -
		# these are static across all pages
		st.title("Tweet Classifer")
		st.subheader(f"Climate change tweet classification using {model}")

	

	elif selection == "Give Me Some Insights...":
        # EDA Page
		st.title("Exploratory Data Analysis: Visualizing the Data")

		st.markdown("Here you'll find graphical insights discovered during the exploratory data analysis phase of the project.")

		# Add image paths for your EDA graphs
		eda_image_path_1 = "resources/imgs/unbalanced_dist_pie.png"
		eda_image_path_2 = "resources/imgs/distribution_of_sentiment_pie.png"
		eda_image_path_3 = "resources/imgs/2_wordcloud.png"
		eda_image_path_4 = "resources/imgs/1_wordcloud.png"
		eda_image_path_5 = "resources/imgs/0_wordcloud.png"
		eda_image_path_6 = "resources/imgs/-1_wordcloud.png"
		eda_image_path_7 = "resources/imgs/2_hashtags.png"
		eda_image_path_8 = "resources/imgs/1-hashtags.png"
		eda_image_path_9 = "resources/imgs/0_hashtags.png"
		eda_image_path_10 = "resources/imgs/-1_hashtags.png"

		# Display the EDA images with headings and explanations
		st.subheader("Graph 1 - Distribution of Sentiments When Unbalanced:")
		st.image(eda_image_path_1, use_column_width=True)
		st.markdown("""
		   		This Pie chart shows the distribution of the raw data we had. As you can see a high majority of the data are 'Pro' tweets. 
		   		This is a problem because the model will be biased towards predicting 'Pro' tweets. To solve this problem we downsampled the data 
				to have an more equal distribution of the sentiments without sacrificing too much data.
					""")

		st.subheader("Graph 2 - Distribution of Sentiments When Downsampled:")
		st.image(eda_image_path_2, use_column_width=True)
		st.markdown("""
		   		This Pie chart shows the distribution of the downsampled data. As you can see the data is now has a slightly better distribution, and we didnt lose
			  too much data! 
					""")
		
		st.subheader("Graph 3 - Wordcloud for Class 'FACT' :")
		st.image(eda_image_path_3, use_column_width=True)
		st.markdown("""
		   		This wordcloud shows the most common words in the 'FACT' class. As you can see the most common word is 'https'. It seems that majority of the "FACT" tweets
			  are links to articles. 
					""")
		
		st.subheader("Graph 4 - Wordcloud for Class 'PRO' :")
		st.image(eda_image_path_4, use_column_width=True)
		st.markdown("""
		   		This wordcloud shows the most common words in the 'PRO' class. 
					""")
		
		st.subheader("Graph 5 - Wordcloud for Class 'NEUTRAL' :")
		st.image(eda_image_path_5, use_column_width=True)
		st.markdown("""
		   		This wordcloud shows the most common words in the 'NEUTRAL' class.
					""")
		
		st.subheader("Graph 7 - Top Hashtags and their Distributions for class 'FACT':")
		st.image(eda_image_path_7, use_column_width=True)
		st.markdown("""
		   		This distibution plot shows the top 10 hashtags and their distributions for the 'FACT' class. 
			  Interestingly, some of the most common Hashtags are related to political climate events. Such as #COP22, which is the
			  2016 United Nations Climate Change Conference, an international meeting of political leaders and activists to discuss environmental issues. 
					""")
		
		st.subheader("Graph 8 -  Top Hashtags and their Distributions for class 'PRO':")
		st.image(eda_image_path_8, use_column_width=True)
		st.markdown("""
		   		This distibution plot shows the top 10 hashtags and their distributions for the 'PRO' class. There are also frequent mentions of political climate events, 
			  such as #COP22 and #ParisAgreement.
					""")
		
		st.subheader("Graph 9 - Top Hashtags and their Distributions for class 'NEUTRAL' :")
		st.image(eda_image_path_9, use_column_width=True)
		st.markdown("""
		   		This distibution plot shows the top 10 hashtags and their distributions for the 'NEUTRAL' class.
					""")
		
		st.subheader("Graph 10 - Top Hashtags and their Distributions for class 'ANTI' :")
		st.image(eda_image_path_10, use_column_width=True)
		st.markdown("""
		   		This distibution plot shows the top 10 hashtags and their distributions for the 'ANTI' class. A standout hashtag to note is #ClimateScam and #OpChemtrails.
			  These hashtags are associated with conspiracy theories that climate change is a hoax. #OPChemtrails is a hashtag used by conspiracy theorists to spread the idea that
			  the government is using airplanes to spray chemicals into the air to control the weather.
					""")
	
	elif selection == "About the Models":
        # Model Info Page
		st.title("How the Models Work")
		st.image("resources/imgs/intro_image.webp", use_column_width=True)

		st.markdown("""
			  Here you'll find information about the models used in the app.

			  - We have employed 3 choices of models for the app: Logistic Regression, SGD Classifier, and Support Vector Classifier.

			  **Brief Brief Explanation of How These Models Work**:
	
			 **Support Vector Classifier (SVM):**
			- *How it works:* SVM finds a hyperplane in a high-dimensional space that separates data into classes. It works well for both linear and non-linear classification.
			- *Key Idea:* It focuses on finding the optimal boundary (hyperplane) that maximally separates different classes.
			  
			 **Logistic Regression:**
			- *How it works:* Logistic Regression models the probability of an instance belonging to a particular class using a logistic function.
			- *Key Idea:* It's a linear model that predicts the probability of binary or multi-class outcomes.
			  
			 **Stochastic Gradient Descent (SGD):**
			- *How it works:* SGD optimizes a linear model by iteratively adjusting weights using a small random subset of the training data.
			- *Key Idea:* It combines many weak models to create a strong predictive model.
			  """)

	

	# Building out the predication page
	if selection == "Let's Classify!":
		st.info("Enter your tweet to classify:")

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

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main(raw)
