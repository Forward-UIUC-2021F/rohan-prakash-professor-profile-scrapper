'''
This Function is the core of the module and should reduce total information size for a given mode(e.g 'research', 'bio')
from all data_packets in data_store.

Currently doing a simple merge by concating them all to a single string, and deliminating a break in data packets with a ' >< '.
'''
# nltk tokenizer for sentence split

import scipy
from sentence_transformers import SentenceTransformer

def sentenceSimilariy(data,compareData):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    corpus=[i for i in data.split('\n')if i != ''and len(i.split(' '))>=4]

    corpus_embeddings = model.encode(corpus)
    query_embeddings = model.encode(compareData)


    # For each search term return 5 closest sentences
    closest_n = 5
    for query, query_embedding in zip(compareData, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n======================\n\n")
        print("Query:", query)

        for idx, distance in results[0:closest_n]:
            print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))

def merge_information(data_store, mode):
    ret_info = ""

    for data_packet in data_store:
        ret_info+=data_packet[mode]
        ret_info+=" >< "
    return ret_info


def consolidate_data(data_store):
    if len(data_store) == 0:
        return {}
    
    modes = data_store[0].keys()
    ret_data_packet = {i : "" for i in modes}

    for m in modes:
        ret_data_packet[m] = merge_information(data_store=data_store,mode=m)

    return ret_data_packet


if __name__ == '__main__':
    _c="""
        Coronavirus:
        White House organizing program to slash development time for coronavirus vaccine by as much as eight months (Bloomberg)
        Trump says he is pushing FDA to approve emergency-use authorization for Gilead's remdesivir (WSJ)
        AstraZeneca to make an experimental coronavirus vaccine developed by Oxford University (Bloomberg)
        Reopening:
        Inconsistent patchwork of state, local and business decision-making on reopening raising concerns about a second wave of the coronavirus (Politico)
        White House risks backlash with coronavirus optimism if cases flare up again (The Hill)
        Florida plans to start reopening on Monday with restaurants and retail in most areas allowed to resume business in most areas (Bloomberg)
        California Governor Newsom plans to order closure of all state beaches and parks starting Friday due to concerns about overcrowding (CNN)
        Japan preparing to extend coronavirus state of emergency, which is scheduled to end 6-May, by about another month (Reuters)
        Policy/Stimulus:
        Economists from a broad range of ideological backgrounds encouraging Congress to keep spending to combat the coronavirus fallout and don't believe now is time to worry about deficit (Politico)
        Global economy:
        China's official PMIs mixed with beat from services and miss from manufacturing (Bloomberg)
        China's Beige Book shows employment situation in Chinese factories worsened in April from end of March, suggesting economy on less solid ground than government data (Bloomberg)
        Japan's March factory output fell at the fastest pace in five months, while retail sales also dropped (Reuters)
        Eurozone economy contracts by 3.8% in Q1, the fastest decline on record (FT)
        US-China:
        Trump says China wants to him to lose his bid for re-election and notes he is looking at different options in terms of consequences for Beijing over the virus (Reuters)
        Senior White House official confident China will meet obligations under trad deal despite fallout from coronavirus pandemic (WSJ)
        Oil:
        Trump administration may announce plans as soon as today to offer loans to oil companies, possibly in exchange for a financial stake (Bloomberg)
        Munchin says Trump administration could allow oil companies to store another several hundred million barrels (NY Times)
        Norway, Europe's biggest oil producer, joins international efforts to cut supply for first time in almost two decades (Bloomberg)
        IEA says coronavirus could drive 6% decline in global energy demand in 2020 (FT)
        Corporate:
        Microsoft reports strong results as shift to more activities online drives growth in areas from cloud-computing to video gams (WSJ)
        Facebook revenue beats expectations and while ad revenue fell sharply in March there have been recent signs of stability (Bloomberg)
        Tesla posts third straight quarterly profit while Musk rants on call about need for lockdowns to be lifted (Bloomberg)
        eBay helped by online shopping surge though classifieds business hurt by closure of car dealerships and lower traffic (WSJ)
        Royal Dutch Shell cuts dividend for first time since World War II and also suspends next tranche of buyback program (Reuters)
        Chesapeake Energy preparing bankruptcy filing and has held discussions with lenders about a ~$1B loan (Reuters)
        Amazon accused by Trump administration of tolerating counterfeit sales, but company says hit politically motivated (WSJ)
        Trump contradicts US intel, says Covid-19 started in Wuhan lab.
        """
    data = 'Kevin C. Chang is a Professor in Computer Science, University of Illinois at Urbana-Champaign . He received a BS from National Taiwan University and PhD from Stanford University , in Electrical Engineering. His research addresses large scale information access, for search, mining, and integration across structured and unstructured big data, with current focuses on "entity-centric" Web search/mining and social media analytics. He received two Best Paper Selections in VLDB 2000 and 2013, an NSF CAREER Award in 2002, an NCSA Faculty Fellow Award in 2003, IBM Faculty Awards in 2004 and 2005, Academy for Entrepreneurial Leadership Faculty Fellow Award in 2008, and the Incomplete List of Excellent Teachers at University of Illinois in 2001, 2004, 2005, 2006, 2010, and 2011. He is passionate to bring research results to the real world and, with his students, co-founded Cazoodle , a startup from the University of Illinois, for deepening vertical "data-aware" search over the web.'


        # 'Bio. Kevin C.C. Chang is a Professor in Computer Science , University of '
        # 'Illinois at Urbana-Champaign , where he leads the FORWARD Data Lab '
        # 'for search, integration, and mining of data. He received a BS from '
        # 'National Taiwan University and PhD from Stanford University , in '
        # 'Electrical Engineering. His research addresses large scale '
        # 'information access, for search, mining, and integration across '
        # 'structured and unstructured big data, with current focuses on '
        # '"entity-centric" Web search/mining and social media analytics. He '
        # 'received two Best Paper Selections in VLDB 2000 and 2013, an NSF '
        # 'CAREER Award in 2002, an NCSA Faculty Fellow Award in 2003, IBM '
        # 'Faculty Awards in 2004 and 2005, Academy for Entrepreneurial '
        # 'Leadership Faculty Fellow Award in 2008, and the Incomplete List of '
        # 'Excellent Teachers at University of Illinois in 2001, 2004, 2005, '
        # '2006, 2010, and 2011. He is passionate to bring research results to '
        # 'the real world and, with his students, co-founded Cazoodle , a '
        # 'startup from the University of Illinois, for deepening vertical '
        # '"data-aware" search over the web. Seung-won Hwang. PhD. Research: '
        # 'Supporting Ranking for Data Retrieval . Aug, 2000 - Jun, 2005. First '
        # 'employment: Assistant Professor, Pohang University of Science and '
        # 'Technology, Pohang, Gyeongbuk, Korea.  ><  >< ',
    compareSentence = ['says Covid-19 started in Wuhan lab', 'lose his bid for re-election and notes he is looking at different options','plans to start reopening on Monday with restaurants and retail']
    
    sentenceSimilariy(_c, compareSentence)