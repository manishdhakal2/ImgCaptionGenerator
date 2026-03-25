

import torch

class Encoder():
    def __init__(self):
        self.max_length = None
        self.pad_index = None
        self.special_indices= []

        
    #----------------------------------------------#

    def build_vocab(self, captions):
        vocab_set = set()

        tokenized_list = [self.tokenize_sentence(sentence) for sentence in captions]

        for sentence in tokenized_list:
            for word in sentence:
                vocab_set.add(word)

        self.vocab_list = [ "<PAD>", "<START>", "<END>", "<UNK>"] + sorted(list(vocab_set)) 

        self.word2idx = {word:idx for idx, word in enumerate(self.vocab_list)}

        self.pad_index = self.word2idx["<PAD>"]

        self.special_indices = [self.word2idx[x] for x in [ "<PAD>", "<START>", "<END>", "<UNK>"] ]

        self.idx2word = {idx:word for idx,word in enumerate(self.vocab_list)}

    
    #----------------------------------------------#

    def tokenize(self,sentence):
        return sentence.lower().split()
    
    #----------------------------------------------#

    def encode_text(self, captions):


        max_length = 0

        tokenized_list = [self.tokenize_sentence(sentence) for sentence in captions]
        
        for sentence in tokenized_list:
            if len(sentence)> max_length:
                max_length = len(sentence)
        
        self.max_length = max_length
        



        encoded_list = []

        for sentence in tokenized_list:


            encoded_sentence =[
                
                self.word2idx["<START>"],
                *[self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sentence], 
                self.word2idx["<END>"]
            ]

            if len(encoded_sentence)<max_length:
                for  i in range(max_length - len(encoded_sentence)):
                    encoded_sentence.append(self.word2idx["<PAD>"])
            encoded_list.append(encoded_sentence)
        
        return encoded_list
    
    #----------------------------------------------#
    
    def decode_text(self, indices):


        decoded_list = []

        for index in indices:

            decoded_sentence =[
                self.idx2word[i] for i in index
            ]
            decoded_list.append(decoded_sentence)

        

        decoded_string_list = []

        
        
        for element in decoded_list:
            final_string = ""
            for index in element:
                if index not in self.special_indices:
                    final_string += f"{self.idx2word[index]} "
            decoded_string_list.append(final_string)
            
        
        return decoded_list


