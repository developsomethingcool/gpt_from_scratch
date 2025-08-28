
from typing import List

class BPETokenizer(special_tokens, tie_break, seed):

    def __init__(self):
        # Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab = {}
        # Maps token_str to token_id (e.g., {"some": 11246})
        self.inverse_vocab = {}
        # Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges = {}

        #  rank dict of form {(string_A, string_B): rank}, where lower rank = higher priority
        self.bpe_ranks = {}
     
    def train(self, text: str, merges: int) -> None:
        pass
     
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
          pass
     
    def decode(self, ids: List[int]) -> str:
          pass

    def save(self, out_dir: str, tag: str) -> None:
          pass

