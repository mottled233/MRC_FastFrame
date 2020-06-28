import paddle.fluid as fluid

from util.util_filepath import *
from util.util_logging import UtilLogging as ULog
from preprocess.tokenizer_CHN import ChnTokenizer as CToken
from preprocess.batching import *
from tqdm import tqdm
from preprocess.preprocess import Preprocess
import random
import collections


class PreprocessForMRCChinese(Preprocess):
    def __init__(self, args, examples, is_prediction=False, cache=""):
        super().__init__(args, examples, is_prediction, cache)
        self.max_seq_length = self.args["max_seq_length"]
        self.tokenizer = CToken(
            self.args["vocab_name"], self.args["vocab_format"], self.args["vocab_type"], self.args["do_lowercase"]
        )
        self.logger = ULog(args, __name__)
        self.logger.info("Successfully build tokenizer")
        # 使用指定的字典，构建tokenizer
        self.batch_size = args['batch_size']

    @staticmethod
    def whitespace_tokenize(text):
        """Runs basic whitespace cleaning and splitting on a peice of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    @staticmethod
    def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                             orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return new_start, new_end

        return input_start, input_end

    @staticmethod
    def _check_is_max_context(doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    def _process_feature(self, **kwargs):
        """
        将examples转化为feature对象
        :return: features
        """
        examples = self.examples
        max_query_length = self.args['max_query_length']
        max_seq_length = self.args['max_seq_length']
        doc_stride = self.args['doc_stride']
        is_training = not self.is_prediction
        tokenizer = self.tokenizer

        features = []
        unique_id = 1000000000
        for (example_index, example) in enumerate(tqdm(examples)):
            query_tokens = tokenizer.tokenize(example['question'])
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example['doc_tokens']):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            # 计算准确的答案开始token位置和结束token位置
            if is_training:
                tok_start_position = orig_to_tok_index[example['start_position']]
                if example['end_position'] < len(example['doc_tokens']) - 1:
                    tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    example['answer'])

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            # 将过长文本分为数段
            doc_spans = []
            _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            # 将每一段转化为一个feature
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[str(len(tokens))] = tok_to_orig_index[split_token_index]
                    is_max_context = self._check_is_max_context(doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[str(len(tokens))] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)
                pos_ids = list(range(0, len(input_ids))) + [0] * (max_seq_length - len(input_ids))
                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(pos_ids) == max_seq_length

                # 映射每一段中的答案开始位置和结束位置
                start_position = None
                end_position = None
                if is_training:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    if tok_start_position == -1 and tok_end_position == -1:
                        start_position = 0  # 问题本来没答案，0是[CLS]的位子
                        end_position = 0
                    else:  # 如果原本是有答案的，那么去除没有答案的feature
                        out_of_span = False
                        doc_start = doc_span.start  # 映射回原文的起点和终点
                        doc_end = doc_span.start + doc_span.length - 1

                        if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
                            out_of_span = True
                        if out_of_span:
                            start_position = 0
                            end_position = 0
                        else:
                            doc_offset = len(query_tokens) + 2
                            start_position = tok_start_position - doc_start + doc_offset
                            end_position = tok_end_position - doc_start + doc_offset

                features.append({'unique_id': unique_id,
                                 'example_index': example_index,
                                 'doc_span_index': doc_span_index,
                                 'tokens': tokens,
                                 'token_to_orig_map': token_to_orig_map,
                                 'token_is_max_context': token_is_max_context,
                                 'input_ids': input_ids,
                                 'input_mask': input_mask,
                                 'segment_ids': segment_ids,
                                 'pos_ids': pos_ids,
                                 'start_position': start_position,
                                 'end_position': end_position})
                unique_id += 1
        print('features num:', len(features))
        return features

    def sample_generator(self):
        """
        返回单个样本的data_generator
        :return:
        """
        self.logger.info("Preprocessing a new round of data of {}".format(len(self.features)))
        if self.args["shuffle"]:
            random.shuffle(self.features)
        for feature in self.features:
            if self.is_prediction:
                yield feature['unique_id'], np.array(feature['input_ids']), np.array(feature['pos_ids']), \
                      np.array(feature['segment_ids']), np.array(feature['input_mask'])

            else:
                yield feature['unique_id'], np.array(feature['input_ids']), np.array(feature['pos_ids']), \
                      np.array(feature['segment_ids']), np.array(feature['input_mask']), \
                      feature['start_position'], feature['end_position']


