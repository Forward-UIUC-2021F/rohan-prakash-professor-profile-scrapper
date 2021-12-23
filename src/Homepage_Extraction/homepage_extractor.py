from numpy.core.fromnumeric import shape
from Homepage_Extraction.parse_data import get_connection, get_raw_data, process_data
import pprint
import pickle

def load_model():
    RF_MODEL = 'rohan-prakash-professor-profile-scrapper/src/Homepage_Extraction/Models/text_classifier'
    SVM_MODEL = 'src/Homepage_Extraction/Models/scikit_learn_sgd'
    SGD_MODEL = 'src/Homepage_Extraction/Models/scikit_learn_svm'
    TCRF_MODEL = ''

    # Load Model
    with open(SVM_MODEL, 'rb') as training_model:
        model = pickle.load(training_model)
    # Load Vectorizer
    with open('src/Homepage_Extraction/Models/vectorizer', 'rb') as vect:
        tfidf_vect = pickle.load(vect)
    return model, tfidf_vect

'''
Go through a page. 
Extract information: 
    -Biography
    -Education
    -Awards
    -Research interests
'''
def extract_homepage(homepage_url):
    DATA_TOPICS = ['edu', 'bio', 'research', 'award']

    data_dic = {x: "" for x in DATA_TOPICS}
    relation_dic = {i+1: DATA_TOPICS[i] for i in range(len(DATA_TOPICS))}
 
    # Create a connection to the url and get HTML
    soup = get_connection(homepage_url)
    # Parse the raw text from HTML
    raw_data = get_raw_data(soup)
    # Do preprocessing on raw data extracted
    data = process_data(raw_data)
 
    # Load both models
    model, vector = load_model()
    # Make predictions
    test_vec = data 
    prediction = model.predict(test_vec)

    res = list(map(int, prediction.tolist()))

    for i in range(len(data)):
        # New curated implementation stores and returns in a dictionary with data criterion/mode as keys
        if res[i] in [1,2,3,4]:
            data_dic[relation_dic[res[i]]] += data[i]+" "

    return data_dic

if __name__ == '__main__':
    test_url = "http://www.forwarddatalab.org/kevinccchang"
    res = extract_homepage(test_url)
    DATA_TOPICS = ['edu', 'bio', 'research', 'award']

    for x in DATA_TOPICS:
        print(x,': \n', res[x])
        print('---------------->\n')
    pprint.pprint(res)