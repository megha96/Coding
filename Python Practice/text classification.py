import nltk
import random #we want random text file not organized
from nltk.corpus import movie_reviews
import pickle

documents = [(list(movie_reviews.words(fileid)),category)#category is pos or neg
             for category in movie_reviews.categories() #each review has a file id
             for fileid in movie_reviews.fileids(category)]

#for category in movie_reviews.categories():
#   for fileid in movie_reviews.fileids(category):
#       documents.append(list(movie_reviews.words(fileid)), category)



random.shuffle(documents)
#we use random to shuffle our documents
#becuz we are going to do training n testing

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())  #first getting all the letters in lower case

all_words = nltk.FreqDist(all_words)

#Now, we are going to list out top 3000 words as features and find whether they
#occur in neg OR pos review
#features[w] will give ans in boolean as to whether the review conatins word or not

word_features = list(all_words.keys())[:3000]
#this list contains 3000 most frequent words

def find_feature(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words) #will give ans in booleans

    return features


#print((find_feature(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_feature(rev),category) for (rev, category) in documents]
    
#find_feature(rev) is same as print argument
#NAIVE BAYES CLASSIFIER- In machine learning, naive
#Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem
#with strong (naive) independence assumptions between the features.
training_set = featuresets[:1900]
test_set = featuresets[1900:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
print("Classifier acuracy percent: ", (nltk.classify.accuracy(classifier,test_set))*100)
classifier.show_most_informative_features(15)

#USAGE OF PICKLE (load and dump)
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

