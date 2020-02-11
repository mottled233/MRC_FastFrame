from preprocess.tokenize_tool import *


class ChnTokenizer(object):
    """
    用以完成对整段文字的tokenize，使用Basic方法完成短语拆分
    """

    def __init__(self, vocab_name="vocab", vocab_format="txt", file_type="vocab", do_lowercase=True):

        self.vocab = load_vocab(vocab_name, vocab_format, file_type)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.text_splitter = TextSplitter(do_lowercase=do_lowercase)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        # vocab: 编码字典
        # inv_vocab: 解码字典

    def tokenize(self, text):
        """
        :param text: str
        :return: list of str
        """

        split_tokens = []
        for token in self.text_splitter.text_split(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):

        return convert_tokens_to_ids(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):

        return convert_ids_to_tokens(self.inv_vocab, ids)


class TextSplitter(object):
    """
    用以完成对完整文本段的拆分，做到对西文特殊字符的处理，并将中日韩文字与标点符号进行单独拆分
    """

    def __init__(self, do_lowercase=True):

        self.do_lowercase = do_lowercase
        # do_lowercase: 是否对大写字母小写处理，包括对西文特殊字符处理

    def text_split(self, text):
        """
        :param text: str
        :return: list of str
        """

        text = convert_to_unicode(text)
        text = clean_text(text)
        text = tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lowercase:
                token = token.lower()
                token = run_strip_accents(token)
            split_tokens.extend(run_split_on_punc(token))
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens
