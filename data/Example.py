class Example(object):
    def __init__(self,
                 qas_id,
                question,
                answer,
                yes_or_no,
                docs,
                docs_selected):
        self.qas_id = qas_id
        self.question = question
        self.answer = answer
        self.yes_or_no = yes_or_no
        self.docs = docs
        self.docs_selected = docs_selected

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question: %s" % (self.question)
        s += ", answer: %s" % (self.answer)
        s += ", yes_or_no: %s" % (self.yes_or_no)
        s += ", len(docs): %d" % (len(self.docs))
        s += ", len(docs_selected): %d" % (len(self.docs_selected))
        return s