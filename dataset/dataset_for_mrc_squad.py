from dataset.dataset import Dataset
from dataset.example import Example
from util.util_filepath import read_file, get_fullurl
import json as js


class DatasetForMrcSquad(Dataset):
    def __init__(self, args):
        super().__init__(args)
        pass

    def _from_srcfile(self, srcfile_path, **kwargs):
        """
            读取squad格式的数据集文件，初始化本身的examples列表
        """

        examples = []
        path = srcfile_path
        is_training = kwargs['is_training']
        assert path != "", "if use read_from_srcfile, must pass in src_file_path when initialize dataset!"
        with open(path, "r", encoding='utf-8') as f:
            train_data = js.load(f)['data']
        mis_match = 0
        for entry in train_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_pos = None
                    end_pos = None
                    orig_answer_text = None
                    if is_training:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        if answer["answer_start"] == -1:
                            answer_offset = paragraph_text.find(orig_answer_text)
                        else:
                            answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        doc_tokens = [paragraph_text[:answer_offset],
                                      paragraph_text[answer_offset: answer_offset + answer_length],
                                      paragraph_text[answer_offset + answer_length:]]
                        start_pos = 1
                        end_pos = 1
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        # actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        actual_text = " ".join(doc_tokens[start_pos:(end_pos + 1)])
                        cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            print("Could not find answer: '%s' vs. '%s'",
                                  actual_text, cleaned_answer_text)
                            continue
                        if actual_text != cleaned_answer_text:
                            # print(actual_text, 'V.S', cleaned_answer_text)
                            mis_match += 1
                            continue
                            # ipdb.set_trace()
                    else:
                        doc_tokens = [paragraph_text]
                    example = Example([('doc_tokens', doc_tokens),
                                       ('qid', qas_id),
                                       ('question', question_text),
                                       ('answer', orig_answer_text),
                                       ('start_position', start_pos),
                                       ('end_position', end_pos),
                                       ])
                    examples.append(example)
        self.logger.info('examples num : {}'.format(len(examples)))
        self.logger.info('mis_match : {}'.format(mis_match))
        return examples


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens
