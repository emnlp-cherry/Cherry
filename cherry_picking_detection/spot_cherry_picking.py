# STEP #5: this script spots cherry-picking by finding important statements that are not mentioned
# in biased articles and generates a score for each article, then combines articles scores per source/outlet.
from sentence_transformers import SentenceTransformer
lang_model = SentenceTransformer('all-MiniLM-L6-v2')
import codecs
from src.helper_scripts.helpers import read_json
from numpy import array, average
from nltk import sent_tokenize
from scipy import spatial
from correlation import *
import string
translation_table = str.maketrans('', '', string.digits)


def curate_controversial_events_by_keyword():
    with codecs.open("event_titles_and_ids_controversial.tsv", 'r', encoding='utf8') as f:
        with codecs.open("event_titles_and_ids_controversial_capitol.tsv", 'w', encoding='utf8') as out:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                event_id, title = line.split("\t")
                if "trump" in title.lower() or "biden" in title.lower():
                    out.write(line+'\n')
            out.close()
        f.close()


def get_controversial_events(events):
    controversial_ids = []
    with codecs.open("event_titles_and_ids_controversial_test.tsv",'r',encoding='utf8') as f:
        with codecs.open("event_titles_and_ids_controversial_test2.tsv",'w',encoding='utf8') as out:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                event_id, title = line.split("\t")
                for event in events:
                    if event["nuetral_articles"][0]["title"]== title:
                        out.write(event_id+'\t'+event["event_id"]+'\t'+title+'\n')
                        controversial_ids.append(event["event_id"])
                        break
            out.close()
        f.close()
    return controversial_ids

def get_cluster_centroid_vector(cluster):
    vectors = []
    for statement in cluster:
        vectors.append(lang_model.encode(statement['text']))
    vecs = array(vectors)
    centroid = average(vecs, axis=0)
    return centroid

def calculate_min_distance_between_article_and_vector(vector,articles):
    distances = []
    statements = []
    for article in articles:
        sentences = sent_tokenize(article)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:   # exclude sentences whose length is less than 10 characters
                sent_vector = lang_model.encode(sent)
                distance = 1 - spatial.distance.cosine(sent_vector, vector)
                distances.append(distance)
                statements.append(sent)

    index_of_min = distances.index(min(distances))
    most_similar_statement = statements[index_of_min]
    return min(distances),most_similar_statement

def exclude_close_statements_from_missing_using_centroids(missing_statements,events, threshold):
    for event_id, missings in missing_statements.items():
        print("Excluding similar statements for event : "+event_id)
        for outlet, missing_states in missings.items():
            new_missing_states = []
            for s in missing_states:
                missing_cluster_id = s[1]  # get the ID of the missing cluster
                for event in events:       # get the whole cluster
                    if event["event_id"] == event_id:   # get the event using its id to extract the whole articles in the event
                        cluster= event["grouped_statements"][missing_cluster_id]
                        cluster_centroid = get_cluster_centroid_vector(cluster["statements"])  # get the vector of the centroid of this cluster
                        #cluster_centroid = lang_model.encode(cluster["statements"][0]["text"])
                        outlet_articles = []                             # get all the articles from this outlet that covered this event
                        for article in event["left_articles"]:
                            if article["outlet"] ==outlet:
                                outlet_articles.append(article["text"])
                        for article in event["right_articles"]:
                            if article["outlet"] ==outlet:
                                outlet_articles.append(article["text"])
                        for article in event["nuetral_articles"]:
                            if article["outlet"] ==outlet:
                                outlet_articles.append(article["text"])

                        min_distance, most_similar_sent = calculate_min_distance_between_article_and_vector(cluster_centroid, outlet_articles)  # calculate the min distance between the centroid of the cluster and all the statements in the articles covered by this outlet
                        if min_distance > threshold:                                                                         # if the min distance is larger than tthe threshold, then there is no single sentence in the article close to the missing statements in the cluster
                            new_missing_states.append(tuple((cluster["statements"][0]["text"],missing_cluster_id)))          # in this case, add this as a certainly missing statement
                        #else:
                            #print("Cluster statement:", cluster["statements"][0]['text'])
                            #print("Most similar sentence: ", most_similar_sent)

            missing_statements[event_id][outlet]= new_missing_states
    return missing_statements


def spot_missing_statements(events_with_predictions_json,distance_threshold,exclude_close_statements):
    data = read_json(events_with_predictions_json)
    events  = data['events']#[:600]
    missing_statements = dict()
    #controversial_events = get_controversial_events(events)
    for event in events:
        #if not event["event_id"] in controversial_events:
        #    continue
        event_covering_sources = dict()
        for article in event['left_articles']:   # getting all sources that cover the event
            if not article['outlet'] in event_covering_sources:
                event_covering_sources[article['outlet']]=[]
        for article in event['right_articles']:
            if not article['outlet'] in event_covering_sources:
                event_covering_sources[article['outlet']]=[]
        for article in event['nuetral_articles']:
            if not article['outlet'] in event_covering_sources:
                event_covering_sources[article['outlet']] = []

        for cluster_id, cluster in event["grouped_statements"].items():
            if cluster["importance"]=="1" and int(cluster_id)>-1:  # if the statement is important and does not belong to cluster -1
                statement_covering_sources=[]
                for statement in cluster["statements"]:
                    if not statement["outlet"] in statement_covering_sources:
                        statement_covering_sources.append(statement["outlet"])
                for source, missing_stat in event_covering_sources.items():  # check for the existence of every outlet in the cluster, if it does not exist, add the missing statement to the outlet in the dictionary
                    if not source in statement_covering_sources:
                        event_covering_sources[source].append(tuple((cluster["statements"][0]["text"],cluster_id)))

        missing_statements[event["event_id"]]=event_covering_sources

    if exclude_close_statements:
        missing_statements = exclude_close_statements_from_missing_using_centroids(missing_statements, events, threshold=distance_threshold)

    missing_avg_by_outlet = dict()    # calculate average number of missing important statements per article
    for event_id, covering_outlets in missing_statements.items():
        for outlet, missings in covering_outlets.items():
            if not outlet in missing_avg_by_outlet:
                missing_avg_by_outlet[outlet] = dict()
                missing_avg_by_outlet[outlet]["missing_sum"]=0
                missing_avg_by_outlet[outlet]["total_events"] = 0
                missing_avg_by_outlet[outlet]["missing_avg"] = 0
            missing_avg_by_outlet[outlet]["missing_sum"] +=len(missings)
            missing_avg_by_outlet[outlet]["total_events"] +=1
            missing_avg_by_outlet[outlet]["missing_avg"] = missing_avg_by_outlet[outlet]["missing_sum"]/missing_avg_by_outlet[outlet]["total_events"]

    print(missing_avg_by_outlet)

    return missing_statements, missing_avg_by_outlet


def read_mbfc_scores(mbfc_scores_file):
    mbfc_scores= dict()
    with codecs.open(mbfc_scores_file,'r',encoding='utf8') as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            line= line.strip()
            outlet, scores = line.split(":")
            bias_label, bias_score, factuality_label, factuality_score =  scores.split(",")
            mbfc_scores[outlet]= dict()
            mbfc_scores[outlet]["bias_score"] = bias_score
            mbfc_scores[outlet]["factuality_score"] = factuality_score
        f.close()
    return mbfc_scores


def combine_scores_by_outlet(missing_avg_scores, mbfc_scores):
    all_scores =dict()
    for outlet, scores in mbfc_scores.items():
        if outlet in missing_avg_scores:
            all_scores[outlet]= dict()
            all_scores[outlet]["external_bias"] = scores["bias_score"]
            all_scores[outlet]["cherry_picking"] = missing_avg_scores[outlet]["missing_avg"]
    return all_scores



curate_controversial_events_by_keyword()


missing_statements, missing_avg_score = spot_missing_statements("bias_analysis_events_clustered_wpredictions.json",distance_threshold=0.1,exclude_close_statements=False)
external_bias_scores = read_mbfc_scores("mbfc_scores2.txt")  # or allSides_scores.txt
all_scores = combine_scores_by_outlet(missing_avg_score, external_bias_scores)
external_bias =[]
mbfc_factuality = []
cherry_picking = []
for outlet, scores in all_scores.items():
    external_bias.append(float(scores["external_bias"]))
    cherry_picking.append(scores["cherry_picking"])



calculate_correlation(external_bias, cherry_picking, "external bias", "cherry-picking")
visualize_pearson_correlation(external_bias, cherry_picking, "external bias", "cherry-picking")


# calculate the average of cherry-picking bias grouped by bias classificaiton from MBFC
grouped_by_bias={"left":[],"left center":[],"right":[],"right center":[],"least biased":[]}
with codecs.open("mbfc_scores2.txt", 'r', encoding='utf8') as f:
    f.readline()
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        outlet, scores = line.split(":")
        bias_label, bias_score, factuality_label, factuality_score = scores.split(",")
        bias_label = bias_label.translate(translation_table)
        bias_label=bias_label.strip()
        outlet_cherrypicking_score = all_scores[outlet]['cherry_picking']
        grouped_by_bias[bias_label].append(outlet_cherrypicking_score)
    f.close()

for bias_lebel, scores in grouped_by_bias.items():
    mean = sum(scores) / len(scores)
    variance = sum([((x - mean) ** 2) for x in scores]) / len(scores)
    res = variance ** 0.5
    print("Avg cherry-picking score for "+bias_lebel +" bias = "+str(mean))
    print("STD = "+str(res))