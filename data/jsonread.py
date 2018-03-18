import json
import os
import numpy as np
import pickle

file_directory = './bible_databases-master/t_kjv.json'
json_data=open(file_directory)

data = json.load(json_data)['resultset']['row']

def format_data(data):
    data_length = len(data)
    formatted_data = []
    index = 0
    book = 0
    chapter = 0
    while index < data_length:
        entry  = data[index]['field']
        phrase = entry[4] 
        
        if entry[1] != book:
            formatted_data.append([])
            book = entry[1]
            chapter = entry[2]
            formatted_data[book-1].append(phrase)
        elif entry[2] != chapter:
            formatted_data[book-1].append(phrase)
            chapter = entry[2]
        else:
            formatted_data[book-1][chapter-1] += ' ' + phrase

        index += 1
    return formatted_data

def save_file(data, filename):
    save_file = open(filename, 'wb')
    pickle.dump(data, save_file)
    save_file.close()

data = format_data(data)
save_file(data, 'bible.txt')
