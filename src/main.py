from Homepage_Finding.homepage_finder import homepage_finder
from Homepage_Extraction.homepage_extractor import extract_homepage
from Data_Consolidator.consolidate_data import consolidate_data
import pprint

def search_professor(prof_tuple):
    data_store = []
    # Module 1
    professor_page_list = homepage_finder(prof_tuple)
    print(professor_page_list, "\n")
    # Module 2
    # Latest Iteration works to use TCRF model for extract_homepage function
    for url in professor_page_list:
        data_store.append(extract_homepage(url))
    # Module 3
    db_data_dic = consolidate_data(data_store=data_store)

    return db_data_dic

if __name__ == '__main__':

    while True:
        print("\n",)
        p_name = input("Enter Professor Name: ")
        if p_name == "q":
            exit()
        p_institution = input("Enter Professor Institution: ")

        print("Data for ", p_name, '\n')
        test_tup = (p_name, p_institution)

        # This single line runs the entire pipeline. Goes from scrapping to classifying to merging and returns.
        db_data = search_professor(test_tup)

        pprint.pprint(db_data)
        # for dic in data_store:
        #     pprint.pprint(dic)
        print("---------------WEBSITE END-----------------")
