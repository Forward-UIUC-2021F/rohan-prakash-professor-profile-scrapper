# professor-profile-scrapper

A Conditional Random Field based web profile scrapper for U.S. Research professors in the field of Computer Science.


## Setup
1) Install necessary module dependencies
```
pip install -r requirements
```
2) Download `TCRF_TrainingData.zip` from [Forward Shared Data Drive](https://drive.google.com/drive/folders/1aovAiLgNLKRCc9QMergSt24yW9ko0ivG). Uncompress the zip file into a folder named `data/`


To find professor profile information, use command line to search a professor.
### Using as a command-line utility
1) We will run main.py and this will begin a command-line based search.

2) This search will pull at most 3 urls related to professor and scrap them

3) Finally it will present the data merged together, each is labeled with category of data (ex. Bio, awards)


## Project Structure
```
rohan-prakash-professor-profile-scrapper/
  - requirements.txt
  - data/
    - 898 labeled_profile .txt files
  - src/
    - main.py
    - webProfiling.py
    - homepage_finding/
        - homepage_finder.py
    - homepage_extraction
        - Models/
            - scikit_learn_sgd
            - scikit_learn_svm
            - text_classifier
            - vectorizer
        - corpus
        - homepage_extractor.py
        - parse_data.py
        - random_forest.py
        - train_classifier.py
        - Test/
            - test_url_bank.txt
    - data_consolidator/
        - consolidate_data.py
```

* `data/898_files`: This folder of 900 labeled profile webpages is used to train TCRF model for homepage extraction
* `src/`: Contains all module code for homepage finding, exraction, and consolidation; focus on extraction for this iteration.
    * `src/main.py`: The driver for the professor search and scrapping pipeline. Connects modules together into search pipeline.
    *  `src/webProfiling.py`: This should be moved into src/homepage_extraction once a model may be trained and generated for use. This is CRF based extraction implementation attempt.
    * `src/homepage_finding/homepage_finder.py`: individual module function that filters google search for at most three professor homepages.
    * `src/homepage_extraction/`: This contains multiple modeling approaches to web profile extraction problem. Models/ contains few models used during development, TCRF model should be placed here once trained.
    * `src/data_consolidator/consolidate_data.py`: Individual module function that aims to merge data extracted from each professor homepages mined. In progress, halted at using sentence similarities.

## Functional Design 

### HomepageFinding
* Finds the top 3 webpage search results for a given professor query, filters to find best homepages.

```python
homepage_finder(query=(professor_name, institution)):
  ...
  return [homepage1_url, homepage2_url, homepage_3_url]
```
### HomepageExtraction
* Extracts information from a given homepage and works to classify them into 4 data categories and an unrelated category. Leverages ML models, latest must use TCRF model.

```python
extract_homepage(homepage_url):
  ...
  return {'edu':..., 'bio':...,'research':..., 'award':...}
```

### DataConsolidator
* Aims to merge data packets(dictionaries) from each homepage into a single data packet. Looks to reduce redundancy currently based on sentence similarities.

```python
consolidate_data(data_store={'edu':..., 'bio':...,'research':..., 'award':...}):
  ...
  return final_data_packet={'edu':..., 'bio':...,'research':..., 'award':...}
```

## Algorithmic Design

Given a search query of a professor and their current institution, we can find some of their most fundamental biographical information from reputable online sources. The key is to find reputable facts about a professor and his interests. We thus turn to modern NLP models to determine what is usefule information and how to classify it into categories.

This process is split up into a three-staged pipeline.

We begin by looking for possible homepage urls on google, these are likely to be top results. Thus we filter based on reputable website url domains (ex. .edu). We cap the number of urls to 3 as too many increases redundancy and volatility of factual information.

Following this each url is then scrapped and mined for pertenant information that fits into 4 data categories. This is where nlp models are used, in particular the TCRF works the most effectively as it takes into account the html tree structure that most homepage's fit into. Once data it is extracted it moves onto consolidation.

In the final stage we aim to reduce overlapping information for better data management. This is currently carried out through sentence similarty using SentenceTransformers. This can be improved as it is a trivial module at current stage.

## References

Some useful references stem from 'A Combination Approach to Web User Profiling'

- http://keg.cs.tsinghua.edu.cn/jietang/publications/TKDD11-Tang-et-al-web-user-profiling.pdf

They have suggested the use of TCRF for this problem and linked aminer.org as a very similar problem, where they used the TCRF model. Below is a webpage used to store the training data for the model used.

- https://www.aminer.org/lab-datasets/profiling/index.html
