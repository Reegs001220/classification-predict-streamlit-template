###  How it works

This process requires the user to input text
(ideally a tweet relating to climate change), and will
classify whether the text or 'Tweet" is factual, supports, opposes, or falls in between the belief that 
climate change is man-made and real. Below you will find information about the data source
and a brief data description. 

###  Data Description

The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo.

This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. Each tweet is labelled independently by 3 reviewers. This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded).

Each Tweet is labelled as one of the following classes:

- 2(News): The Tweet links to factual news about climate change
- 1(Pro): The Tweet supports the belief of man-made climate change
- 0(Neutral): The Tweet neither supports nor refutes the belief of man-made climate change
- -1(Anti): The Tweet does not believe in man-made climate change

### Future Reccomendations

Our models can only improve the more data we get. We reccomend that more Tweets from current posts are added to the Dataframe, specifically tweets that represent the minority classes, such as classes 2, 0 and -1. The more our model can practice, the better it will learn and the better we can classify your inputs. 