import json
import os
import numpy as np
import pickle

file_directory = './t_kjv.json'
json_data=open(file_directory)
data = json.load(json_data)['resultset']['row']

def format_data(data):
    data_length = len(data)
    formatted_data = ''
    index = 1
    book = 0
    chapter = 0

    entry  = data[0]['field']
    phrase = entry[4] 
    formatted_data = phrase
    while index < data_length:
        entry  = data[index]['field']
        phrase = entry[4] 
        
        """
        if entry[1] != book:
            book = entry[1]
            chapter = entry[2]
            formatted_data.append(phrase)
        elif entry[2] != chapter:
            formatted_data.append(phrase)
            chapter = entry[2]
        else:
            formatted_data[-1] += ' ' + phrase
        """

        formatted_data += ' ' + phrase
        index += 1
    return formatted_data

def save_file(data, filename):
    save_file = open(filename, 'wb')
    pickle.dump(data, save_file)
    save_file.close()

data = format_data(data)
print(data[0])
print(len(data))
save_file(data, 'corpus.pkl')
