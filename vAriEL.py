#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:18:53 2019

vAriEL for New Word Acquisition

0. if you want to put it on keras you need to use numbers and not words
     - maybe create a module inside vAriEL that transforms a sentences 
     generator into a numbers generator
1. character level
2. lose complete connection to grammar
3. every node the same number of tokens
4. start with a number of tokens larger than necessary, and 
     - assign tokens to characters them upon visit, first come first served

"""

import os
import logging
import numpy as np
from nltk import CFG
import scipy.linalg
import pickle
from decimal import Decimal, localcontext, Context

from nlp import GrammarLanguageModel, Vocabulary, addEndTokenToGrammar, \
                NltkGrammarSampler, tokenize

from AriEL import SentenceEncoder, SentenceDecoder, SentenceEmbedding



class vAriEL_Encoder(SentenceEncoder):

    INTERNAL_PRECISION = 100
    EOS = '<EOS>'

    def __init__(self, grammar, ndim=1, precision=np.finfo(np.float32).precision, dtype=np.float32, transform=None):
        super(vAriEL_Encoder, self).__init__()
        self.__dict__.update(grammar=grammar, ndim=ndim, precision=precision, dtype=dtype, transform=transform)
        self.languageModel = GrammarLanguageModel(addEndTokenToGrammar(grammar, INTERNAL_PRECISION.EOS))

    def encode(self, sentence):

        with localcontext(Context(prec=vAriEL_Encoder.INTERNAL_PRECISION)):
            # Initial range
            lowerBounds = [Decimal(0)] * self.ndim
            upperBounds = [Decimal(1)] * self.ndim
            curDim = 0

            self.languageModel.reset()
            for token in tokenize(sentence) + [vAriEL_Encoder.EOS]:
                # Get the distribution for the next possible tokens
                nextTokens = sorted(list(self.languageModel.nextPossibleTokens()))
                if token not in nextTokens:
                    raise Exception('Input sentence does not respect the grammar! : ' + sentence)
                probs = 1.0 / len(nextTokens) * np.ones(len(nextTokens))

                # CDF range
                cdfLow = np.concatenate([[0.0], np.cumsum(probs)[:-1]])
                cdfHigh = np.cumsum(probs)

                # Update range from the CDF range of given token
                idx = nextTokens.index(token)
                curRange = upperBounds[curDim] - lowerBounds[curDim]
                print(upperBounds)
                print(lowerBounds)
                print(token)
                print(nextTokens)
                print('')
                upperBounds[curDim] = lowerBounds[curDim] + (curRange * Decimal(cdfHigh[idx]))
                lowerBounds[curDim] = lowerBounds[curDim] + (curRange * Decimal(cdfLow[idx]))

                self.languageModel.addToken(token)

                # NOTE: at each iteration, change the dimension
                curDim += 1
                if curDim >= self.ndim:
                    curDim = 0

            # Compute the middle point of the final range
            z = [lowerBound + ((upperBound - lowerBound) / Decimal(2))
                 for lowerBound, upperBound in zip(lowerBounds, upperBounds)]
            z = np.array(z, dtype=self.dtype)

        if self.transform is not None:
            z = np.dot(z, self.transform)

        z = np.round(z, self.precision)

        return z

class vAriEL(SentenceEmbedding):

    def __init__(self, grammar, 
                 ndim=1, 
                 precision=np.finfo(np.float32).precision, 
                 dtype=np.float32, 
                 nbMaxTokens=100, 
                 transform=None, 
                 flexibleBounds=False,
                 name='embedding_arithmetic'):
        self.name = name
        
        if isinstance(grammar, list):
            self.grammarOutputAgent = grammar[1]
            self.grammarInputAgent = grammar[0]
        else:
            self.grammarOutputAgent = grammar
            self.grammarInputAgent = grammar
            
        vocabulary = Vocabulary.fromGrammar(self.grammarInputAgent)
        super(ArithmeticCodingEmbedding, self).__init__(vocabulary, ndim)

        if precision > np.finfo(dtype).precision:
            logger.warning('Reducing precision because it is higher than what %s can support (%d > %d): ' % (str(dtype), precision, np.finfo(dtype).precision))
            precision = np.finfo(dtype).precision
        self.__dict__.update(grammar=grammar, precision=precision, dtype=dtype, nbMaxTokens=nbMaxTokens)

        # FIXME: I simplified the code to keep it mentally manageable
        self.transform = None
        self.transformInv = None
            
        # define encoder and decoder
        self.encoder = vAriEL_Encoder(self.grammarInputAgent, ndim=ndim, precision=precision, dtype=dtype, transform=self.transform)
        self.decoder = vAriEL_Decoder(self.grammarOutputAgent, ndim=ndim, precision=precision, transform=self.transformInv)

    def getEncoder(self):
        return self.encoder

    def getDecoder(self, languageModel = None):
        return self.decoder





# grammar cannot have recursion!
grammar = CFG.fromstring("""
                         S -> NP VP | NP V
                         VP -> V NP
                         PP -> P NP
                         NP -> Det N
                         Det -> 'a' | 'the'
                         N -> 'dog' | 'cat'
                         V -> 'chased' | 'sat'
                         P -> 'on' | 'in'
                         """)


    

def test_flexibleBounds():
    sentence = 'the dog chased the dog'
    embedding = AriEL(grammar, ndim = 2, precision = 10, flexibleBounds=True)
    encoder = embedding.getEncoder()
    print(encoder.encode(sentence))
    
    # ideally I want to be able to write sth like:
    
    #vAriEL = vAriEL(grammarModel)
    
    #model = Sequential()
    
    #model.add(vAriEL.encoder())
    #model.add(Dense())
    #model.add(vArieL.decoder())
    
    #model.fit_generator(_generator(grammar), samples_per_epoch=10, nb_epoch=10)
    
    
    
    

if __name__ == '__main__':
    #test_flexibleBounds()
    
