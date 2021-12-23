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
data = open('rohan-prakash-professor-profile-scrapper/src/Homepage_Extraction/corpus').read()

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
