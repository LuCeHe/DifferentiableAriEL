import os
import logging
import numpy as np
from nltk import CFG
import scipy.linalg
import pickle
from decimal import Decimal, localcontext, Context

from nlp import GrammarLanguageModel, Vocabulary, addEndTokenToGrammar, \
                NltkGrammarSampler, tokenize




logger = logging.getLogger(__name__)

class SentenceEmbedding(object):

    def __init__(self, vocabulary, latentDim):
        self.__dict__.update(vocabulary=vocabulary, latentDim=latentDim)
        self._init_args = (vocabulary, latentDim)

    def __getinitargs__(self):
        return self._init_args

    def getEncoder(self):
        raise NotImplementedError()

    def getDecoder(self):
        raise NotImplementedError()

    def interpolate(self, sentence1, sentence2, nbSamples=10, removeDuplicates=False):
        # Encode sentences
        encoder = self.getEncoder()
        code1 = np.array(encoder.encode(sentence1))
        code2 = np.array(encoder.encode(sentence2))

        interpolations = []
        decoder = self.getDecoder()
        for x in np.linspace(0, 1, nbSamples):
            c = x * code2 + (1 - x) * code1
            sentence = decoder.decode(c)
            if (not removeDuplicates or
                len(interpolations) == 0 or
                    (len(interpolations) > 0 and sentence != interpolations[-1])):
                interpolations.append(sentence)
        return interpolations

    def coverage(self, grammar, stochastic=False, nbSamples=None):

        encoder = self.getEncoder()
        decoder = self.getDecoder()

        nbValid = 0
        nbTotal = 0

        if stochastic:
            # Stochastic sampling of the grammar
            if nbSamples is None:
                raise Exception('The number of samples must be specified in stochastic sampling mode!')
            sampler = NltkGrammarSampler(grammar)
            for _ in range(nbSamples):
                for sentence in sampler.generate(1):
                    code = encoder.encode(sentence)
                    recons = decoder.decode(code)
                    if recons == sentence:
                        nbValid += 1
                    nbTotal += 1
        else:
            # Depth-first search of the grammar
            from nltk.parse.generate import generate
            for tokens in generate(grammar, n=nbSamples):
                sentence = ' '.join(tokens)
                code = encoder.encode(sentence)
                recons = decoder.decode(code)
                if recons == sentence:
                    nbValid += 1
                nbTotal += 1

        coverage = float(nbValid) / nbTotal
        return coverage

    def save(self, filename):
        logger.info('Saving embedding to file: %s' % (os.path.abspath(filename)))
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        logger.info('Loading embedding from file: %s' % (os.path.abspath(filename)))
        with open(filename, 'rb') as f:
            return pickle.load(f)


class SentenceEncoder(object):

    def encode(self, sentence):
        raise NotImplementedError()


class SentenceDecoder(object):

    def decode(self, z):
        raise NotImplementedError()

class ArithmeticCodingEncoder(SentenceEncoder):

    INTERNAL_PRECISION = 100
    EOS = '<EOS>'

    def __init__(self, grammar, ndim=1, precision=np.finfo(np.float32).precision, dtype=np.float32, transform=None):
        super(ArithmeticCodingEncoder, self).__init__()
        self.__dict__.update(grammar=grammar, ndim=ndim, precision=precision, dtype=dtype, transform=transform)
        self.languageModel = GrammarLanguageModel(addEndTokenToGrammar(grammar, ArithmeticCodingEncoder.EOS))

    def encode(self, sentence):

        with localcontext(Context(prec=ArithmeticCodingEncoder.INTERNAL_PRECISION)):
            # Initial range
            lowerBounds = [Decimal(0)] * self.ndim
            upperBounds = [Decimal(1)] * self.ndim
            curDim = 0

            self.languageModel.reset()
            for token in tokenize(sentence) + [ArithmeticCodingEncoder.EOS]:
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




class DecodingOutOfRange(Exception):
    pass


class DecodingNotEnoughPrecision(Exception):
    pass


class ArithmeticCodingDecoder(SentenceDecoder):

    INTERNAL_PRECISION = 100
    EOS = '<EOS>'

    def __init__(self, grammar, ndim=1, precision=np.finfo(np.float32).precision, nbMaxTokens=100, transform=None, ignoreOutOfRange=False):
        super(ArithmeticCodingDecoder, self).__init__()
        self.__dict__.update(grammar=grammar, ndim=ndim, precision=precision, nbMaxTokens=nbMaxTokens,
                             transform=transform, ignoreOutOfRange=ignoreOutOfRange)
        self.languageModel = GrammarLanguageModel(addEndTokenToGrammar(grammar, ArithmeticCodingDecoder.EOS))

    def decode(self, z):
        with localcontext(Context(prec=ArithmeticCodingEncoder.INTERNAL_PRECISION)):

            if self.transform is not None:
                z = np.dot(z, self.transform)

            if not ((z >= 0.0).all() and (z <= 1.0).all()):
                raise DecodingOutOfRange('The code vector is not in the interval [0,1]: ' + str(z))

            z = [Decimal(float(d)) for d in z]

            # Initial range
            lowerBounds = [Decimal(0)] * self.ndim
            upperBounds = [Decimal(1)] * self.ndim
            curDim = 0

            tokens = []
            token = None
            self.languageModel.reset()
            while token != ArithmeticCodingDecoder.EOS and len(tokens) < self.nbMaxTokens:
                # Get the distribution for the next possible tokens
                if token is not None:
                    self.languageModel.addToken(token)
                nextTokens = sorted(list(self.languageModel.nextPossibleTokens()))
                probs = 1.0 / len(nextTokens) * np.ones(len(nextTokens))

                # CDF range
                cdfLow = np.concatenate([[0.0], np.cumsum(probs)[:-1]])
                cdfHigh = np.concatenate([np.cumsum(probs)[:-1], [1.0]])

                token = None
                for i in range(len(nextTokens)):
                    # Update range from the CDF range of given token
                    curRange = upperBounds[curDim] - lowerBounds[curDim]
                    upperBoundToken = lowerBounds[curDim] + (curRange * Decimal(cdfHigh[i]))
                    lowerBoundToken = lowerBounds[curDim] + (curRange * Decimal(cdfLow[i]))
                    if lowerBoundToken <= z[curDim] <= upperBoundToken:
                        token = nextTokens[i]
                        lowerBounds[curDim] = lowerBoundToken
                        upperBounds[curDim] = upperBoundToken
                        break
                if token is None:
                    if z[curDim] < lowerBounds[curDim] and self.ignoreOutOfRange:
                        token = nextTokens[0]
                    elif z[curDim] >= upperBounds[curDim] and self.ignoreOutOfRange:
                        token = nextTokens[-1]
                    else:
                        raise DecodingOutOfRange('Unable to decode token %s! Try augmenting the number of units' % (token))
                tokens.append(token)

                # NOTE: at each iteration, change the dimension
                curDim += 1
                if curDim >= self.ndim:
                    curDim = 0

        if tokens[-1] != ArithmeticCodingDecoder.EOS:
            raise DecodingNotEnoughPrecision('Unable to properly decode the sentence: adjust the number of dimensions and precision!')
        sentence = ' '.join(tokens[:-1])
        
            
        return sentence


class AriEL(SentenceEmbedding):

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

        if transform == 'orthonormal':
            Q, _ = np.linalg.qr(np.random.randn(ndim, ndim), mode='reduced')
            self.transform = Q.T.astype(dtype=dtype)
            self.transformInv = scipy.linalg.pinv(self.transform).astype(dtype=dtype)
            assert np.allclose(self.transform.dot(self.transform.T), np.identity(self.transform.shape[0]), atol=1e-6)
            assert np.allclose(np.linalg.norm(self.transform, axis=1), np.ones(self.transform.shape[0]), atol=1e-6)
            assert np.allclose(self.transform.dot(self.transformInv), np.identity(self.transform.shape[0]), atol=1e-6)
        elif transform == 'orthogonal':
            w = np.random.randn(ndim, ndim)
            self.transform = np.real(w.dot(scipy.linalg.pinv(scipy.linalg.sqrtm(w.T.dot(w))))).astype(dtype=dtype)
            self.transformInv = scipy.linalg.pinv(self.transform).astype(dtype=dtype)
            assert np.allclose(self.transform.dot(self.transform.T), np.identity(self.transform.shape[0]), atol=1e-6)
            assert np.allclose(self.transform.dot(self.transformInv), np.identity(self.transform.shape[0]), atol=1e-6)
        elif transform == 'random-gaussian':
            self.transformInv = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(ndim), size=(ndim, ndim)).astype(dtype=dtype)
            self.transform = scipy.linalg.pinv(self.transformInv).astype(dtype=dtype)
            assert np.allclose(self.transform.dot(self.transformInv), np.identity(self.transform.shape[0]), atol=1e-4)
        else:
            self.transform = None
            self.transformInv = None
            
        if flexibleBounds:
            self.encoder = ArithmeticCodingEncoder_wRNN(self.grammarInputAgent, ndim=ndim, precision=precision, dtype=dtype, transform=self.transform)
            self.decoder = ArithmeticCodingDecoder(self.grammarOutputAgent, ndim=ndim, precision=precision, transform=self.transformInv)
        else:
            self.encoder = ArithmeticCodingEncoder(self.grammarInputAgent, ndim=ndim, precision=precision, dtype=dtype, transform=self.transform)
            self.decoder = ArithmeticCodingDecoder(self.grammarOutputAgent, ndim=ndim, precision=precision, transform=self.transformInv)

    def getEncoder(self):
        return self.encoder

    def getDecoder(self, languageModel = None):
        return self.decoder





# grammar cannot have recursion!
grammar = CFG.fromstring("""
                         S -> NP VP
                         VP -> V NP
                         PP -> P NP
                         NP -> Det N
                         Det -> 'a' | 'the'
                         N -> 'dog' | 'cat'
                         V -> 'chased' | 'sat'
                         P -> 'on' | 'in'
                         """)

def test_naiveBounds():
    sentence = 'the dog chased the dog'
    embedding = ArithmeticCodingEmbedding(grammar, ndim = 2, precision = 10)
    encoder = embedding.getEncoder()
    print(encoder.encode(sentence))
    

    
    

if __name__ == '__main__':
    test_naiveBounds()