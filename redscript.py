import praw
import pandas as pd
import datetime
from collections import Counter
import re
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
import json

pronouns = ['I', 'you', 'he', 'she', 'it', 'we', 'they',
            'mine', 'yours', 'his', 'hers', 'its', 'ours', 'theirs',
            'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves',
            'this', 'that', 'these', 'those',
            'who', 'whom', 'whose', 'which', 'what',
            'who', 'whom', 'whose', 'which', 'that',
            'all', 'another', 'any', 'anybody', 'anyone', 'anything', 'each', 'everybody', 'everyone', 'everything',
            'nobody', 'none', 'no one', 'nothing', 'one', 'other', 'somebody', 'someone', 'something', 'several',
            'some', 'few', 'many', 'all', 'any', 'both', 'more', 'most', 'none', 'some', 'such']


function_words = ['a', 'an', 'the',
                  'aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'around', 'as',
                  'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'by', 'concerning',
                  'considering', 'despite', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into', 'like',
                  'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past', 'regarding', 'round', 'since',
                  'through', 'throughout', 'to', 'toward', 'under', 'underneath', 'until', 'unto', 'up', 'upon', 'with',
                  'within', 'without',
                  'and', 'but', 'or', 'nor', 'for', 'so', 'yet',
                  'be', 'have', 'do', 'will', 'shall', 'can', 'could', 'may', 'might', 'must', 'should', 'would',
                  'I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'you', 'him', 'her', 'it', 'us', 'them',
                  'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']


def import_connection_data(file_name):
    with open(file_name, 'r') as file:
        conf = json.load(file)
    return conf

def scrape_data(file_name):
    conf= import_connection_data(file_name)
    reddit = praw.Reddit(client_id=conf['client_id'],
                        client_secret=conf['client_secret'],
                        user_agent=conf['user_agent'],
                        username=conf['username'],
                        password=conf['password'])
    subreddit = reddit.subreddit('pikmin')
    posts = subreddit.top('month', limit=1000)

    data = []
    for post in posts:
        data.append({
            'title': post.title,
            'score': post.score,
            #'url': post.url,
            'created_utc': datetime.datetime.fromtimestamp(post.created_utc),
            'flair': post.link_flair_text,
            'Selftext': post.selftext,
            'Author': post.author
        })
    return data


def export_data_csv(data, file_name):
    data.to_csv(file_name, index=False)


def import_data_csv(file_name):
    df = pd.read_csv(file_name)
    return df


def extract_words(text):
    words = re.findall(r'\w+', text.lower())
    return words


def word_count_to_csv(file_name,data) :

    # File path
    file_path = file_name + '.csv'
    # Writing data to CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def clean_data(data, column):
    #remove Nan Authors
    nan_indices = data[data[column].isna()].index.tolist()
    for index_value in nan_indices:
        data.drop(index_value, inplace=True)
    return data


def get_word_counts(data, column_name, limiter=10):
    filter = ['to', 'is', 'my', 'how', 'was', 'up', 'are', 'have', 'just', 'if', 'has', 'when', 'had', 'an']
    filter.extend(pronouns)
    filter.extend(function_words)
    data['words'] = data[column_name].apply(extract_words)
    all_words = []  # Create an empty list to store all the words
    for sublist in data['words']:
        for word in sublist:
            if len(word) < 2 or word in filter :
                continue        
            all_words.append(word)
    word_counts = Counter(all_words)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    word_count_list= []
    for word, count in sorted_words:
        if count < limiter:
            break
        word_count_list.append([word, count])
    return word_count_list


def user_name_count(data, column_name):
    data = clean_data(data,column_name)
    return get_word_counts(data, column_name, 1)

def get_phrases(data, column_name):
    data = clean_data(data,column_name)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data[column_name])
    similarity_matrix = cosine_similarity(vectors)

    threshold = 0.8
    similar_phrases = []

    for i in range(len(data)):
        for j in range(i+1, len(data)):
            if similarity_matrix[i, j] >= threshold:
                similar_phrases.append((data[column_name][i], data[column_name][j]))

    for phrase1, phrase2 in similar_phrases:
        if phrase1.lower() == phrase2.lower():
            continue
        print(f"Similar phrases: {phrase1} | {phrase2}")

analysis_choices=['user_freq','word_counts','similar_phrases']

parser = argparse.ArgumentParser(description='Description of your program')

# Add arguments
parser.add_argument('-s', '--scrape_data', action='store_true', default=False, help='connect to Reddit app and ingest post data')
parser.add_argument('-o', '--output_filename', type=str, default='output', help='')
parser.add_argument('-e', '--export_filename', type=str, default='exported_data', help='')
parser.add_argument('-i', '--input_filename', type=str, default='input', help='previously exported csv to use as dataset')
parser.add_argument('-t', '--analysis_type', choices=analysis_choices, default='word_counts', help='Analysis type choices user_freq, word_counts, similar_phrases. Default=word_counts.')
parser.add_argument('-c', '--conf_filename', type=str, help='Used to set up connection to redit app account')
# Parse the command-line arguments
args = parser.parse_args()

if args.scrape_data:
    if not args.conf_filename or not os.path.exists(args.conf_filename):
        print('conf file not found or provided')
        parser.print_help()
        exit(1)
    else:
        print('Using ' + args.conf_filename)
    df = pd.DataFrame(scrape_data(args.conf_filename))
    export_data_csv(df, args.export_filename)
elif args.analysis_type:
    if not os.path.exists(args.input_filename):
        print('Input file not found')
        exit(1)
    else:
        print('Using ' + args.input_filename)

    df = import_data_csv(args.input_filename)

    if args.analysis_type == 'user_freq':
        words = user_name_count(df, 'Author')
        word_count_to_csv(args.output_filename , words)
    elif args.analysis_type == 'word_counts':
        words = get_word_counts(df, 'title')
        word_count_to_csv(args.output_filename , words)
    elif args.analysis_type == 'similar_phrases':
        get_phrases(df, 'title')
    
else:
    parser.print_help()
