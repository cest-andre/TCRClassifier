import pandas
import numpy as np
import torch
from model import TCRModel, clf_loss_func

'''
:param file_path: path to csv file
:return: tuple of tensors (data, label)
'''
def process_data(file_path):
    df = pandas.read_csv(file_path)
    dictionary = ['0', '1', '<cls>', '<seq>']
    additional_tokens = ['<cls>', '<seq>', None]

    # Find the max length of each column
    maxes = df.applymap(lambda x: len(str(x))).max()

    for column in df.columns:
        
        # Constructing the dictionary
        for string in df[column]:     
            for char in str(string):
                if char not in dictionary:
                    dictionary.append(char)
        
        # Adding padding
        df[column] = df[column].astype(str).str.pad(width=maxes[df.columns.get_loc(column)], side='right', fillchar='0')

        # Split into tokens
        df[column] = [
            [char for char in str(string)]
            for string in df[column]
        ]
        
        # Adding additional tokens
        if additional_tokens[df.columns.get_loc(column)]:
            df[column] = df[column].apply(lambda x: [additional_tokens[df.columns.get_loc(column)]] + x)

        # Numericize the tokens
        df[column] = [
            [dictionary.index(token) for token in string]
            for string in df[column]
        ]

    # Creating the combined sequence
    df['sequence'] = df['antigen']+ df['TCR']

    return torch.tensor(df['sequence']), torch.tensor(df['interaction'])

class Classifier():
    def __init__(self):
        self.model = TCRModel()
        self.loss_func = clf_loss_func

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)

    def train(self, x):
        '''
        Train the model on one batch of data
        :param x: train data
        :return: (mean) loss of the model on the batch
        '''
        return

if __name__ == "__main__":
    data, labels = process_data("./data.csv")
