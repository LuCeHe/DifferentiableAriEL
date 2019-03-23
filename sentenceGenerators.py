# Copyright (c) 2018, 
#
# authors: Luca Celotti
# during their PhD at Universite' de Sherbrooke
# under the supervision of professor Jean Rouat
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.




from nltk import CFG
from nlp import Vocabulary, NltkGrammarSampler
from nltk.parse.generate import generate
import string
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# grammar cannot have recursion!
grammar = CFG.fromstring("""
                         S -> NP VP | NP V
                         VP -> V NP
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
    def __init__(self, grammar, batch_size = 5, maxlen=None, categorical=False):
        
        self.grammar = grammar
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.categorical = categorical
        
        tempVocabulary = Vocabulary.fromGrammar(grammar)
        tokens = tempVocabulary.tokens[1:]
        tokens = set(sorted(' '.join(tokens)))
        self.vocabulary = Vocabulary(tokens)
        #self.tokens = sorted(list(string.printable))
        #self.vocabulary = Vocabulary(self.tokens)
        print('')
        print(self.vocabulary.indicesByTokens)
        print('')
        # +1 to take into account padding
        self.vocabSize = self.vocabulary.getMaxVocabularySize()
        self.startId = self.vocabulary.tokenToIndex(self.vocabulary.endToken)
        
        #self.nltk_generate = generate(self.grammar, n = self.batch_size)
        self.sampler = NltkGrammarSampler(self.grammar)
        
        
    def generator(self, offset=0):
        while True:
            sentences = [[''.join(sentence)] for sentence in self.sampler.generate(self.batch_size)]
            sentencesCharacters = sentencesToCharacters(sentences)
            # offset=1 to take into account padding
            sentencesIndices = [self.vocabulary.tokensToIndices(listOfTokens, offset=offset) for listOfTokens in sentencesCharacters]
            indices = pad_sequences(sentencesIndices, maxlen=self.maxlen)
            
            if not self.categorical:
                yield indices, indices
            else:
                categorical_indices = to_categorical(indices, num_classes = self.vocabSize)
                yield indices, categorical_indices
            
    def indicesToSentences(self, indices, offset=0):
        if not isinstance(indices[0][0], int):
            indices =  [[int(i) for i in list_idx] for list_idx in indices]
        return self.vocabulary.indicesToSentences(indices, offset=offset)
        

    
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
    categorical = False
    generator_class = c2n_generator(grammar, maxlen=20, categorical=categorical)
    generator = generator_class.generator()
    
    print(generator_class.vocabSize)
    
    indicess = next(generator)
    # if categorical
    #if categorical: argmax etc...
    sentences = indicess
    #sentences = generator_class.indicesToSentences(indicess)
    
    print('')
    for indices, sentence in zip(indicess, sentences): 
        print(indices)
        #print(sentence)
        print('')
    print('')
    print('')
    print(len(indicess.shape))
    #(5, 20, 13) if categorical
    #(5, 20)
    
if __name__ == '__main__':
    #test_generator(grammar)
    #test_sentencesToCharacters()    
    test_generator_class()