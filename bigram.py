'''
This file has the implement of bi-gram algorithm, based on the input_ids of LLMs. 
'''
import numpy as np
import torch

class bigram():
    def __init__(self, vocabulary_size:int):
        '''
        Inputs:
            vocabulary_size: the length of vocabulary
        '''
        self._bigram_matrix = np.zeros((vocabulary_size, vocabulary_size), dtype=float)
        
        pass

    def fit(self, input_ids: torch.tensor=None):
        '''
        Inputs:
            input_ids: the 1 dimentional vector input_ids. 
        Return:
            a matrix, with each raw as a vector representing a bi-gram distribution of a given token id.
        '''
        # multi-threads?
        if input_ids:
            for i in range(len(input_ids) - 1):
                self._bigram_matrix[input_ids[i]][input_ids[i + 1]] += 1
            self._bigram_matrix = self._bigram_matrix / ((self._bigram_matrix.sum(axis=1) + 1e-5)[:, np.newaxis])
        else:
            self._bigram_matrix = self._bigram_matrix / ((self._bigram_matrix.sum(axis=1) + 1e-5)[:, np.newaxis])

        return self._bigram_matrix

    def add_and_count(self, input_ids: torch.tensor):
        '''
        Accumulating.
        Combined with fit().
        '''
        for i in range(len(input_ids) - 1):
            self._bigram_matrix[input_ids[i]][input_ids[i + 1]] += 1
    


# # test
# bigram_instance = bigram(13)

# # text = [1,2,3,1,1,2,12,12,1,2,1,2,2,1]
# # result = bigram_instance.fit(text)

# text = [1,2,3,1,1,2,12]
# text2 = [12,12,1,2,1,2,2,1]
# bigram_instance.add_and_count(text)
# bigram_instance.add_and_count(text2)
# result = bigram_instance.fit()

# for e in result:
#     print(e)

# print(np.sum(result, axis=0))
# print(np.sum(result, axis=1))



