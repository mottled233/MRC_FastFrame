from preprocess.tokenize_tool import *
import paddlehub as hub


class EngnieerTokenizer(object):

    def __init__(self, vocab_name="vocab", vocab_format="txt", file_type="vocab", do_lowercase=True):
        self.vocab = load_vocab(vocab_name, vocab_format, file_type)
        self.text_splitter = TextSplitter()
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.lac = hub.Module(name="lac")
        self.engnieer_index = {'n': 4, 'f': 5, 's': 6, 't': 7, 'nr': 8, 'ns': 9, 'nt': 10, 'nw': 11, 'nz': 12, 'v': 13,
                               'vd': 14, 'vn': 15, 'a': 16, 'ad': 17, 'an': 18, 'd': 19, 'm': 20, 'q': 21, 'r': 22,
                               'p': 23, 'c': 24, 'u': 25, 'xc': 26, 'w': 27, 'PER': 28, 'LOC': 29, 'ORG': 30, 'TIME': 31}

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
        ids = [0] * len(tokens)
        text = ''.join(tokens)
        inputs = {"text": [text]}
        results = self.lac.lexical_analysis(data=inputs)
        result = results[0]
        words = result['word']
        tags = result['tag']
        index2tag = ['p'] * len(text)
        index = 0
        # bert的词表分词和lac的不相同，用下标到词性的索引列表来解决
        for lac_index, word in enumerate(words):
            for i in range(index, index + len(word)):
                index2tag[i] = tags[lac_index]
            index = index + len(word)
        index = 0
        for token_index, token in enumerate(tokens):
            tag = index2tag[index]
            if tag in self.engnieer_index:
                encode = self.engnieer_index[tag]
                ids[token_index] = encode
            index += len(token)
        return ids

    def entity_sim(self, text1, text2):
        inputs = {"text": [text1, text2]}
        results = self.lac.lexical_analysis(data=inputs)
        entity_tag = ['n', 'nr', 'ns', 'nt', 'nw', 'nz', 't', 'PER', 'LOC', 'ORG', 'TIME']
        entity1, entity2 = set(), set()
        for index, result in enumerate(results):
            if index == 0:
                for i, tag in enumerate(result['tag']):
                    if tag in entity_tag:
                        entity1.add(result['word'][i])
            if index == 1:
                for i, tag in enumerate(result['tag']):
                    if tag in entity_tag:
                        entity2.add(result['word'][i])
        if len(entity1 & entity2) != 0:
            return True
        return False


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
