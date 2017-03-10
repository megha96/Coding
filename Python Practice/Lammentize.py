#Lammentizing - similar to stemming  but stemming results in meaning less words
#but Lammentizing results in meaning full words
from nltk.stem import WordNetLemmatizer

lammentizer = WordNetLemmatizer()
print(lammentizer.lemmatize("cars"))
print(lammentizer.lemmatize("better" , pos="a")) #default is noun
print(lammentizer.lemmatize("best" , pos="a"))
