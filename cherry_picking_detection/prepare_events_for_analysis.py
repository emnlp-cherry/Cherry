#STEP1: this script prepares the data

from src.helper_scripts.helpers import read_json
import json
import codecs
import os
from nltk import sent_tokenize


def get_ids(event_IDs_file):
    ids = []
    with codecs.open(event_IDs_file, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            fields= line.split("\t")
            ids.append(fields[0])
        f.close()
    return ids

# call this function once to brwose the list of availabel events and select a subset for analysis
def get_event_titles_and_ids(data_files_dir):   # step 1
    selected_events = []
    filenames= os.listdir(data_files_dir)
    ids = get_ids("/home/ijaradat/PycharmProjects/cherry_picking_data/data/input/event_selection.txt")
    with codecs.open("event_titles_and_ids.tsv",'w',encoding='utf8') as out:
        for filename in filenames:
            if filename.endswith(".json"):
                data = read_json(data_files_dir+filename)
                events = data['events']
                for event in events:
                    if not event["id"] in ids and not event["id"] in selected_events:   # to avoid doing analysis on events used in training the models
                        selected_events.append(event["id"])
                        neutral_article = event["neutral_articles"][0]
                        title= neutral_article["title"]
                        out.write(event["id"]+"\t"+title+'\n')
        out.close()


def create_event_and_statement_pool(data_files_dir,selected_events):   # step 2
    filenames= os.listdir(data_files_dir)
    ids = get_ids(selected_events)
    analysis_events = {}
    analysis_events["events"] = []
    for filename in filenames:
        if filename.endswith(".json"):
            data = read_json(data_files_dir+filename)
            events = data['events']
            for event in events:
                analysis_event = dict()
                if event["id"] in ids:
                    neutral_article = event["neutral_articles"][0]
                    analysis_event["context"]  = neutral_article["text"]
                    analysis_event["statement_pool"] =[]
                    left_articles = event['left_articles']
                    right_articles = event['right_articles']
                    analysis_event["left_articles"]= event['left_articles']
                    analysis_event["right_articles"] = event["right_articles"]
                    analysis_event["nuetral_articles"] = event["neutral_articles"]
                    for article in left_articles:
                        sentences = sent_tokenize(article['text'])
                        for sentence in sentences:
                            s= dict()
                            s['text'] = sentence
                            s['outlet'] = article['outlet']
                            s['doc_id'] = article['doc_id']
                            analysis_event["statement_pool"].append(s)
                    for article in right_articles:
                        sentences = sent_tokenize(article['text'])
                        for sentence in sentences:
                            s = dict()
                            if len(sentence.strip())>10:
                                s['text'] = sentence
                                s['outlet'] = article['outlet']
                                s['doc_id'] = article['doc_id']
                                analysis_event["statement_pool"].append(s)

                    analysis_events['events'].append(analysis_event)
    with open("bias_analysis_events.json", 'w', encoding='utf-8') as out:
        json.dump(analysis_events, out, ensure_ascii=False, indent=4)
        out.close()




get_event_titles_and_ids("step_c_output/")
create_event_and_statement_pool("step_c_output/","event_titles_and_ids.tsv")
