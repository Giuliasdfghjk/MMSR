import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

# ================================
# All 4 Retrieval Methods 
# ================================

def find_song_info(*args, info_df):
    if len(args) == 1:
        song_id = args[0]
        song_info = info_df[info_df['id'] == song_id]

        if not song_info.empty:
            return song_info.iloc[0]['artist'], song_info.iloc[0]['song']
        else:
            raise ValueError(f"ID {song_id} not in the dataset.")

    elif len(args) == 2:
        artist, song = args
        matching_songs = info_df[(info_df['artist'] == artist) & (info_df['song'] == song)]

        if not matching_songs.empty:
            return matching_songs.iloc[0]['id']
        else:
            raise ValueError(f"Song {song} by {artist} not in the dataset.")
            return
        return
    return

def music_retrieval(artist, song, df, info_df, N, genre_df):
    try:
        query_id = find_song_info(artist, song, info_df=info_df) 
        query_vector = df[df['id'] == query_id].iloc[0, 1:].values
    except IndexError:
        print('Song ID not found!')
        return
    
    all_songs = df.iloc[:, 1:].values  
    similarities = cosine_similarity([query_vector], all_songs).flatten()
    
    top_indices = similarities.argsort()[-N-1:-1][::-1]
    top_similar_songs = df.iloc[top_indices].copy() 
    top_similar_songs['similarity'] = similarities[top_indices]  

   
    top_similar_songs['artist'], top_similar_songs['song'] = zip(*top_similar_songs['id'].apply(lambda x: find_song_info(x, info_df=info_df)))
    top_similar_songs = pd.merge(top_similar_songs, genre_df, on='id', how='left')

    column_order = ['id', 'artist', 'song', 'genre', 'similarity']
    return top_similar_songs[column_order]


def omni_retriever(retrieval_method, query_song, info_df, genre_df, show=True,N=10, create_csv=True, m_name="mfcc"):
    retrieved_song = music_retrieval(query_song['artist'], query_song['song'], retrieval_method, info_df, N, genre_df)
    if show:
        print("-"*30)
        print(f"Method {m_name} - {N} recommendations for: {query_song['song']}")
        if len(list(query_song.keys())) > 2:
            print("Genre: ", query_song['genre'])
        print("-"*30)
        display(retrieved_song.style.hide(axis="index"))
    if create_csv:
        retrieved_song.to_csv(f"{m_name}.csv", mode='a', header=False, index=False)
    return retrieved_song






# ================================
# Evaluation
# ================================

##### PrecisionRecall
def check_relevance(retrieved_genres, query_genres):
    """Needed for the recall and precision function"""
    retrieved_genres_set = set(eval(retrieved_genres))  
    query_genres_set = set(query_genres)
    return len(retrieved_genres_set.intersection(query_genres_set)) > 0 


def calculate_average_precision_at_k(data, k):
    grouped_data = data.groupby('query_n')

    total_precision = 0
    num_queries = 0

    for query_name, group in grouped_data:
        if len(group) < k:
            raise ValueError(f"Not enough results for query {query_name} to calculate precision at {k}")
        precision_at_k = group.head(k)['relevant'].sum() / k
        total_precision += precision_at_k
        num_queries += 1

    average_precision = total_precision / num_queries if num_queries > 0 else 0
    return average_precision

def calculate_average_recall_at_k(retrieved_df, full_dataset, query_genres, k=10):
    total_recall = 0
    num_queries = 0

    for query_name, genres in query_genres.items():
        query_data = retrieved_df[retrieved_df['query_n'] == query_name]

        if len(query_data) < k:
            raise ValueError(f"Not enough results for query {query_name} to calculate recall at {k}")

        top_k_retrieved = query_data.head(k)
        full_dataset['relevant'] = full_dataset['genre'].apply(lambda x: check_relevance(x, genres))
        total_relevant = full_dataset['relevant'].sum()
        relevant_retrieved = top_k_retrieved['relevant'].sum()
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
        total_recall += recall
        num_queries += 1
        
    average_recall = total_recall / num_queries if num_queries > 0 else 0
    return average_recall

def compute_precision_recall_for_k_values(data, max_k):

    precision_recall_values = {}

    for k in range(1, max_k + 1):
        precision = data.head(k)['relevant'].sum() / k
        recall = data.head(k)['relevant'].sum() / data['relevant'].sum() if data['relevant'].sum() > 0 else 0
        precision_recall_values[k] = (precision, recall)

    return precision_recall_values




#### nDCG10

def sorenson_dice_coefficient(set1, set2):
    if not set1 or not set2:
        return 0.0
    return 2 * len(set1.intersection(set2)) / (len(set1) + len(set2))

def calculate_nDCG_at_10(df, full_df, query_genres):
    DCG = 0
    IDCG = 0

    for index, row in df.head(10).iterrows():
        relevance = sorenson_dice_coefficient(set(eval(row['genre'])), set(query_genres))
        DCG += relevance / np.log2(index + 2)  

    ideal_relevances = []
    for index, row in full_df.iterrows():
        relevance = sorenson_dice_coefficient(set(eval(row['genre'])), set(query_genres))
        ideal_relevances.append(relevance)
    ideal_relevances.sort(reverse=True)

    for i in range(min(10, len(ideal_relevances))):
        IDCG += ideal_relevances[i] / np.log2(i + 2)

    nDCG = DCG / IDCG if IDCG > 0 else 0
    return nDCG

def calculate_average_nDCG_at_10(df, full_df, query_genres):
    total_nDCG = 0
    num_queries = 0

    for query_label, query_genre_set in query_genres.items():
        query_df = df[df['query_n'] == query_label]
        nDCG = calculate_nDCG_at_10(query_df, full_df, query_genre_set)
        total_nDCG += nDCG
        num_queries += 1

    average_nDCG = total_nDCG / num_queries if num_queries > 0 else 0
    return average_nDCG

#### genre coverage
def calculate_genre_coverage_at_10(df):
    all_unique_genres = set()
    for query_index in range(1, 4):  
        query_genres = df[df['query_n'] == f'query_{query_index}']['genre']
        for genres in query_genres:
            all_unique_genres.update(eval(genres))

    covered_genres = set()
    for query_index in range(1, 4):  
        top_10 = df[df['query_n'] == f'query_{query_index}'].head(10)
        for genres in top_10['genre']:
            covered_genres.update(eval(genres))
    genre_coverage = len(covered_genres) / len(all_unique_genres) if all_unique_genres else 0
    return genre_coverage

#### genre_diversity@10

def calculate_genre_diversity_at_10(dataframe):
    unique_genres = set()
    for genres in dataframe['genre']:
        unique_genres.update(genres)
    unique_genres = list(unique_genres)
    genre_diversity_sum = 0.0

    for query_index in range(3):  
        genre_distribution_sum = [0.0] * len(unique_genres)

        for row_index, row in dataframe.iloc[query_index*10:(query_index+1)*10].iterrows():
            genres = row['genre']

            for genre in genres:
                genre_index = unique_genres.index(genre)
                genre_distribution_sum[genre_index] += 1 / len(genres)

        normalized_distribution = [count / 10 for count in genre_distribution_sum]
        genre_diversity = -sum(p * math.log2(p) if p > 0 else 0 for p in normalized_distribution)
        genre_diversity_sum += genre_diversity

    average_genre_diversity = genre_diversity_sum / 3
    return average_genre_diversity

















