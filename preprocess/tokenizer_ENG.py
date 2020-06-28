from preprocess.tokenize_tool import *


class EngTokenizer(object):
    """
    用以完成对整段文字的tokenize，经过字母小写，使用空格拆分
    """

    def __init__(self, vocab_name="vocab", vocab_format="txt", file_type="vocab"):

        self.vocab = load_vocab(vocab_name, vocab_format, file_type)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        """
        :param text: str
        :return: list of str
        """

        split_tokens = []
        for token in text.lower().split(" "):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):

        return convert_tokens_to_ids(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):

        return convert_ids_to_tokens(self.inv_vocab, ids)
