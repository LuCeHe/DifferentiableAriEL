#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 20:07:08 2019

@author: perfect
"""

from nltk import CFG
from nlp import Vocabulary, NltkGrammarSampler
from nltk.parse.generate import generate
import string
from keras.preprocessing.sequence import pad_sequences


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



def _basicGenerator(grammar, batch_size = 3):
    #sentences = []
    while True:
        yield [[' '.join(sentence)] for sentence in generate(grammar, n = batch_size)]


def sentencesToCharacters(sentences):
    """
    The input:
        [['the dog  chased a cat'],
        ['the dog sat in a cat'],
        ['the cat sat on a cat']]
    The output:
        [['t', 'h', 'e', ' ', 'd', 'o', ...],
        ['t', 'h', ...],
        ['t', ...]]
    """
    
    assert isinstance(sentences, list)
    assert isinstance(sentences[0], list)
    assert isinstance(sentences[0][0], str)
    
    charSentences = [list(sentence[0]) for sentence in sentences]
    
    return charSentences
    
    
    
def _charactersGenerator(grammar, batch_size = 5):
    
    # FIXME: the generator is not doing what it should be doing, 
    # since ell the batches are the same
    # but it's fine for now since this is only a toy scenario
    
    while True:
        sentences = [[' '.join(sentence)] for sentence in generate(grammar, n = batch_size)]
        yield sentencesToCharacters(sentences)



def _charactersNumsGenerator(grammar, batch_size = 5):
    
    tokens = sorted(list(string.printable))
    #print(tokens)
    
    vocabulary = Vocabulary(tokens)
    #print(vocabulary.getMaxVocabularySize())
    
    
    # FIXME: the generator is not doing what it should be doing, 
    # since ell the batches are the same
    # but it's fine for now since this is only a toy scenario
    
    while True:
        sentences = [[' '.join(sentence)] for sentence in generate(grammar, n = batch_size)]
        #print(sentences)
        sentencesCharacters = sentencesToCharacters(sentences)
        sentencesIndices = [vocabulary.tokensToIndices(listOfTokens) for listOfTokens in sentencesCharacters]
        padded_indices = pad_sequences(sentencesIndices)
        yield padded_indices
    
    
    
class c2n_generator(object):
    def __init__(self, grammar, batch_size = 5, maxlen=None):
        
        self.grammar = grammar
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.tokens = sorted(list(string.printable))
        self.vocabulary = Vocabulary(self.tokens)
        self.vocabSize = self.vocabulary.getMaxVocabularySize()
        
        #self.nltk_generate = generate(self.grammar, n = self.batch_size)
        self.sampler = NltkGrammarSampler(self.grammar)
        
    def generator(self):
        while True:
            sentences = [[' '.join(sentence)] for sentence in self.sampler.generate(self.batch_size)]
            #print(sentences)
            sentencesCharacters = sentencesToCharacters(sentences)
            sentencesIndices = [self.vocabulary.tokensToIndices(listOfTokens) for listOfTokens in sentencesCharacters]
            padded_indices = pad_sequences(sentencesIndices, maxlen=self.maxlen)
            yield padded_indices
        

    
def test_generator(grammar):
    
    print('Testing basic generator')
    generator = _basicGenerator(grammar)
    print(next(generator))
    print('')
    print(next(generator))
    print('')
    
    
    print('Testing character level generator')
    generator = _charactersGenerator(grammar)
    for sentence in next(generator): print(sentence)
    print('')
    for sentence in next(generator): print(sentence)
    print('')
    
    
    print('Testing character to number generator')
    generator = _charactersNumsGenerator(grammar)
    for sentence in next(generator): print(sentence)
    print('')
    for sentence in next(generator): print(sentence)
    print('')



def test_sentencesToCharacters():
    sentences = [['the dog  chased a cat'],
                 ['the dog sat in a cat'],
                 ['the cat sat on a cat']]
                
    print(sentencesToCharacters(sentences))
    

def test_generator_class():
    
    generator_class = c2n_generator(grammar, maxlen=10)
    generator = generator_class.generator()
    for sentence in next(generator): print(sentence)
    print('')
    for sentence in next(generator): print(sentence)
    print('')
    print('')
    print('')
    
    
    
if __name__ == '__main__':
    #test_generator(grammar)
    #test_sentencesToCharacters()    
    test_generator_class()