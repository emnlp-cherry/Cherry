# STEP #2: This script clusters the statement pool in every event into semantically similar statements, adn generates a json file with
# the clustering information included.
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from src.helper_scripts.helpers import read_json
import numpy as np
import json
lang_model = SentenceTransformer('all-MiniLM-L6-v2')


def cluster_statements(bias_analysis_events_file):
    data = read_json(bias_analysis_events_file)
    events = data['events']
    analysis_events = {}
    analysis_events["events"] = []
    for i,event in enumerate(events):
        print("Clustering statements in event #"+str(i))
        for statement in event['statement_pool']:
            statement['embedding'] = lang_model.encode(statement['text'])   # get sentence embeddings

        vectors = [statement['embedding'] for statement in event['statement_pool']]
        vectors = np.array(vectors)
        model = DBSCAN(eps=0.15, min_samples=2, metric='cosine')
        print("Clustering ...")
        model.fit(vectors)
        print("Clustering ... Done !! ")
        predictions = model.labels_.tolist()
        for i in range(len(event['statement_pool'])):
            event['statement_pool'][i]["cluster_id"] = predictions[i]
            del event['statement_pool'][i]["embedding"]   # removing embeddings to save space and memory

        groups = dict()  # group statements into their clusters
        for s in event['statement_pool']:
            if not s["cluster_id"] in groups:
                groups[s["cluster_id"]] = []
            groups[s["cluster_id"]].append(s)


        event["grouped_statements"] = groups
        analysis_events["events"].append(event)

    with open("bias_analysis_events_clustered.json", 'w', encoding='utf-8') as out:
        json.dump(analysis_events, out, ensure_ascii=False, indent=4)
        out.close()

cluster_statements("bias_analysis_events.json")