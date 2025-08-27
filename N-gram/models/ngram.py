"""N-gram model implementation with required interface - SELF-CONTAINED"""
import pickle
import json
import numpy as np
from collections import defaultdict, Counter
from scipy.special import softmax

class SimpleNGramEngine:
    """Simplified N-gram engine with Laplace smoothing and interpolation"""
    
    def __init__(self, n=3):
        self.n = n
        self.counts = {}
        self.vocab = set()
        self.lambdas = None
        self.total_vocab_size = 0
        
    def fit(self, text_tokens, lambdas=None, pad_token="<s>"):
        """Fit n-gram model on token sequence"""
        if isinstance(text_tokens, str):
            text_tokens = text_tokens.lower().split()
            
        padded_tokens = [pad_token] * (self.n - 1) + text_tokens + ["</s>"]
        self.vocab = set(padded_tokens)
        self.total_vocab_size = len(self.vocab)
        
        self.counts = {}
        for order in range(1, self.n + 1):
            self.counts[order] = Counter()
            for i in range(len(padded_tokens) - order + 1):
                ngram = tuple(padded_tokens[i:i + order])
                self.counts[order][ngram] += 1
                
        if lambdas is None:
            weights = np.linspace(0.1, 1.0, self.n)
            self.lambdas = softmax(weights * 3)
        else:
            self.lambdas = np.array(lambdas)
            
        return self
        
    def get_ngram_prob(self, ngram, interpolate=True):
        """Get probability of n-gram with Laplace smoothing"""
        if isinstance(ngram, str):
            ngram = tuple(ngram.lower().split())
        elif not isinstance(ngram, tuple):
            ngram = tuple(ngram)
            
        if not interpolate:
            count = self.counts[len(ngram)].get(ngram, 0)
            if len(ngram) == 1:
                total = sum(self.counts[1].values())
            else:
                prefix = ngram[:-1]
                total = sum(v for k, v in self.counts[len(prefix)].items() if k == prefix)
            return (count + 1) / (total + self.total_vocab_size)
            
        prob = 0.0
        for order in range(1, min(len(ngram), self.n) + 1):
            suffix = ngram[-order:]
            order_prob = self.get_ngram_prob(suffix, interpolate=False)
            prob += self.lambdas[order - 1] * order_prob
            
        return prob

class NGramModel:
    """N-gram model with DoD-compliant interface"""
    def __init__(self, n=3):
        self.n = n
        self.engine = None
        
    def fit(self, text_data, lambdas=None, pad_token="<s>"):
        """Fit n-gram model on training data"""
        self.engine = SimpleNGramEngine(self.n)
        self.engine.fit(text_data, lambdas, pad_token=pad_token)
        return self
    
    def save(self, filepath):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n': self.n,
                'engine': self.engine
            }, f)
    
    def load(self, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.n = data['n']
            self.engine = data['engine']
        return self
    
    def logprob(self, next_token, context):
        """Get log probability of next_token given context"""
        if self.engine is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(context, str):
            context = context.split()
        if isinstance(next_token, str):
            query = context + [next_token]
        else:
            query = list(context) + [next_token]
        
        prob = self.engine.get_ngram_prob(query, interpolate=True)
        return np.log(max(prob, 1e-10))
