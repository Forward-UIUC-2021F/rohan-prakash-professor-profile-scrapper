import pandas
import numpy as np
import pickle
from sklearn import model_selection, preprocessing, metrics, ensemble
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

# load the dataset
data = open('V2/Homepage_Extraction/corpus').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    if content[-1] not in ['0', '1', '2', '3', '4', '5', '6']:
        print(content[-1])
    labels.append(content[-1])
    texts.append(" ".join(content[:-1]))
print("Total item:", len(texts), len(labels))

# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

# split the dataset into training and validation datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=.2, stratify=trainDF['label'], random_state=42)

def sgd_pipeline():
    return Pipeline(
        [
            (
                "tfidf_vector_com",
                TfidfVectorizer(
                    input="array",
                    norm="l2",
                    token_pattern=r'\w{1,}',
                    max_features=None,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log",
                    penalty="l2",
                    class_weight='balanced',
                    tol=0.001,
                ),
            ),
        ]
    )

def svc_pipleline():
    return Pipeline(
        [
            (
                "tfidf_vector_com",
                TfidfVectorizer(
                    input="array",
                    norm="l2",
                    #token_pattern=r'\w{1,}',
                    max_features=None,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                SVC(
                    C=10,
                    kernel="rbf",
                    gamma=0.1,
                    probability=True,
                    class_weight=None,
                ),
            ),
        ]
    )


def print_metrics(pred_test, y_test, pred_train, y_train):
    print("test accuracy", str(np.mean(pred_test == y_test)))
    print("train accuracy", str(np.mean(pred_train == y_train)))
    print("\n Metrics and Confusion for SVM \n")
    print(metrics.confusion_matrix(y_test, pred_test))
    print(metrics.classification_report(y_test, pred_test))


# # Support Vector Machine Model
svc_pipe = svc_pipleline()
svc_pipe.fit(X_train, y_train)
pred_test = svc_pipe.predict(X_test)
pred_train = svc_pipe.predict(X_train)
print_metrics(pred_test, y_test, pred_train, y_train)



# Stochastic Gradient Descent Model
sgd_pipe = sgd_pipeline()
sgd_pipe.fit(X_train, y_train)
pred_test = sgd_pipe.predict(X_test)
pred_train = sgd_pipe.predict(X_train)
print_metrics(pred_test, y_test, pred_train, y_train)




with open('V2/Homepage_Extraction/Models/scikit_learn_sgd', 'wb') as f:
    pickle.dump(sgd_pipe, f)
    
with open('V2/Homepage_Extraction/Models/scikit_learn_svm', 'wb') as f:
    pickle.dump(svc_pipe, f)




# # label encode the target variable
# encoder = preprocessing.LabelEncoder()
# train_y = encoder.fit_transform(train_y)
# test_y = encoder.fit_transform(test_y)

# # word level tf-idf
# TFIDF = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
# TFIDF.fit(trainDF['text'])
# train_fe = TFIDF.transform(train_x)
# test_fe = TFIDF.transform(test_x)


# # print(TFIDF.vocabulary_)


# def train_model(classifier, feature_vector_train, label, feature_vector_valid):
#     # fit the training dataset on the classifier
#     rf = classifier.fit(feature_vector_train, label)

#     # predict the labels on validation dataset
#     predictions = classifier.predict(feature_vector_valid)

#     return metrics.accuracy_score(predictions, test_y), rf, predictions


# # RF on Word Level TF IDF Vectors
# accuracy, classifier, prediction = train_model(ensemble.RandomForestClassifier(), train_fe, train_y, test_fe)
# print("RF, WordLevel TF-IDF: ", accuracy)

# print(confusion_matrix(prediction, test_y))
# print(classification_report(prediction, test_y))
# print(accuracy_score(prediction, test_y))

# Save classification model and vectorizer model

# with open('Models/text_classifier', 'wb') as picklefile:
#     pickle.dump(classifier, picklefile)

# with open('Models/vectorizer', 'wb') as picklefile:
#     pickle.dump(TFIDF, picklefile)


