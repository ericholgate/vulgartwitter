"""
Run to prepare data set for modeling.
Automatically uses the path ./data/coling_twitter_data.tsv 
to the original data set but if your file path
is different then you can change it using the flag --data_set
Saves the cleaned data to ./data/cleaned_data.tsv
If the flag --split=True, then it will automatically save a train, validation, 
and testing set as separate files. Default is --split=True

Usage example:
python3 clean_data.py --data_set=./data/coling_twitter_data.tsv --split=True

How this program pre-processes code:
(1) Lower cases tweets, removes leading and trailing punctuation for each word
(2) Adds masked_tweet, insert_tweet, and num_vulgar columns
"""

import re
import string
import pandas as pd
import argparse
import copy

from unidecode import unidecode
from collections import OrderedDict


# Reads in file and returns a list of curse words
def read_csv_curse_words():
    curse_words = []
    f = open("noswearingclean.csv", "r")
    translator = str.maketrans('', '', string.punctuation)
    for line in f:
        line = line.strip()
        line = line.split(",")
        word = line[0].translate(translator)
        curse_words.append(word)
    f.close()
    return curse_words


# Binary search for list of words
def find(L, target):
    start = 0
    end = len(L) - 1
    while start <= end:
        middle = (start + end) // 2
        midpoint = L[middle]
        if midpoint > target:
            end = middle - 1
        elif midpoint < target:
            start = middle + 1
        else:
            return True
    return False


# Returns a list of regexes
def read_regex():
    re_str = []

    # Read in the lines from the regex input file.
    with open("regexes.txt", 'r') as f:
        for line in f:
            re_str.append(line.split('\n')[0])
    f.close()

    # From each regex string, create a tuple of the compiled regex and the dictionary-
    # accurate label (i.e., what variants caught by that regex should be treated as)
    regexes = []
    for variant in re_str:
        expr, label = variant.split(",")
        regexes.append((re.compile(expr), label.strip()))
    return regexes


# Checks if word matches a regex
# Returns an empty string for no match and the collapsed word for a match
def check_regex(word, regexes):

    for expr in regexes:
        if bool(re.fullmatch(expr[0], word)):
            return expr[1]
    return ""


if __name__ == "__main__":
    # Data path
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', default="./data/coling_twitter_data.tsv", type=str, action='store')
    parser.add_argument('--split', default=True, type=bool, action='store')
    args = parser.parse_args()
    params = vars(args)

    # Read in file and create data frame
    df = pd.read_csv(params['data_set'], sep='\t')

    # Add columns
    df['masked_tweet'] = ''
    df['insert_tweet'] = ''
    df['num_vulgar'] = 0

    tweets = df["Tweet"]
    n = df.shape[0]

    curse_words = read_csv_curse_words()
    regex_list = read_regex()

    total_rows = df.shape[0]  # Total number of tweets
    swear_dict = {}  # Dictionary of swear words
    count_vulgar_tweets = 0
    for i in range(n):
        tweet = unidecode(tweets[i])
        tweet = tweet.replace("\n", "")  # Remove new lines in tweets
        vulgar = False
        words = tweet.split()
        masked_words = copy.deepcopy(words)
        insert_words = copy.deepcopy(words)
        insert_idx = []

        for j in range(len(words)):
            if words[j] == "<USER>" or words[j] == "<URL>":
                pass
            else:
                stripped_word = words[j].lower().strip(string.punctuation)
                # If word is a dictionary match
                if find(curse_words, stripped_word):
                    vulgar = True

                    # Add swear words to dictionary
                    if stripped_word in swear_dict:
                        swear_dict[stripped_word] += 1
                    else:
                        swear_dict[stripped_word] = 1

                    words[j] = stripped_word
                    insert_words[j] = stripped_word
                    masked_words[j] = '<VG>'
                    insert_idx.append(j)

                else:
                    # If not a dictionary match, check if regex match
                    regex_word = (check_regex(stripped_word, regex_list))
                    if regex_word != "":
                        vulgar = True
                        insert_idx.append(j)
                        if regex_word in swear_dict:
                            swear_dict[regex_word] += 1

                            words[j] = regex_word
                            insert_words[j] = regex_word
                            masked_words[j] = '<VG>'
                        else:
                            swear_dict[regex_word] = 1

                    # If not a regex match, word is not vulgar
                    else:
                        words[j] = stripped_word
                        masked_words[j] = stripped_word
                        insert_words[j] = stripped_word

        num_vulgar = len(insert_idx)

        if vulgar:
            count_vulgar_tweets += 1
            if num_vulgar == 1:
                insert_words.insert(insert_idx[0] + 1, "<VG>")
            else:
                add_idx = 1
                for k in range(num_vulgar):
                    insert_idx[k] += add_idx
                    add_idx += 1
                for k in insert_idx:
                    insert_words.insert(k, "<VG>")
        df.at[i, "Tweet"] = " ".join(words)
        df.at[i, "masked_tweet"] = " ".join(masked_words)
        df.at[i, "insert_tweet"] = " ".join(insert_words)
        df.at[i, "num_vulgar"] = num_vulgar

    word_stats_sorted = OrderedDict(sorted(swear_dict.items(), key=lambda x: x[1], reverse=True))
    word_stats_sorted = list(word_stats_sorted.items())

    print("Total number of vulgar tweets: {}".format(count_vulgar_tweets))
    print("Total number of tweets: {}".format(total_rows))
    print("Percent vulgar: {} %".format(round((count_vulgar_tweets/total_rows) * 100, 3)))
    print("-----------------------------------------------------------------")
    print("Top 50 words")
    for i in range(50):
        print(word_stats_sorted[i][0], ":", word_stats_sorted[i][1])

    df.to_csv('./data/cleaned_data.tsv', sep='\t')

    if params['split']:
        val = df[:500]
        test = df[500:1500]
        train = df[1500:]

        val.to_csv('./data/cleaned_data_val.tsv', sep='\t')
        test.to_csv('./data/cleaned_data_test.tsv', sep='\t')
        train.to_csv('./data/cleaned_data_train.tsv', sep='\t')

