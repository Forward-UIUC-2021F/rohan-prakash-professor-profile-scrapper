# professor-profile-scrapper

A Conditional Random Field based web profile scrapper for U.S. Research professors in the field of Computer Science.


## Setup
1) Install necessary module dependencies
```
pip install -r requirements
```
2) Download `TCRF_TrainingData.zip` from [Forward Shared Data Drive](https://drive.google.com/drive/folders/1aovAiLgNLKRCc9QMergSt24yW9ko0ivG). Uncompress the zip file into a folder named `data/`


To find papers by keyword, use this module either as command-line utility or a library
### Using as a command-line utility
1) We will run data_main.py



## Project Structure
```
rohan-prakash-professor-profile-scrapper/
  - requirements.txt
  - data/
    - 898 labeled_profile .txt files
  - src/
    - find_papers.py
    - store_papers.py
    - find_paper_by_keyword/
      - assign_paper_keywords.py
      - database.py
      - embeddings_generator.py
      - paper_indexer.py
      - rank_papers.py
      - utils.py
    - fild_readers/
      - keyword_file_reader.py
      - paper_file_reader.py
```