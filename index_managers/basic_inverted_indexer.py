import numpy as np


class BasicInvertedIndexer:
    def __init__(self, doc_ids, vocabulary, term_doc_incidence_matrix):
        self.doc_ids = doc_ids
        self.vocabulary = vocabulary
        self.term_doc_incidence_matrix = term_doc_incidence_matrix
        self.index = dict().fromkeys(vocabulary)

    def build_index(self):
        print(self.index.keys())
        for token_index in range(len(self.vocabulary)):
            indexes = dict().fromkeys(['doc_ids'])
            indexes['doc_ids'] = list()
            for doc_index in range(len(self.doc_ids)):
                if self.term_doc_incidence_matrix[doc_index, token_index] == 1:
                    indexes['doc_ids'].append(self.doc_ids[doc_index])
            self.index[self.vocabulary[token_index]] = indexes


if __name__ == '__main__':
    import os
    from doc_parsers.pdf_parser import Pypdf2Parser

    # 步骤一：解析原始文档
    pdf_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    my_pdf_parser = Pypdf2Parser(pdf_dir)
    my_pdf_parser.parse_docs()

    # 步骤二：构建索引
    my_doc_ids = my_pdf_parser.doc_ids
    my_vocabulary = my_pdf_parser.vocabulary
    my_term_doc_incidence_matrix = my_pdf_parser.term_doc_incidence_matrix
    my_basic_inverted_indexer = BasicInvertedIndexer(my_doc_ids, my_vocabulary, my_term_doc_incidence_matrix)
    my_basic_inverted_indexer.build_index()
