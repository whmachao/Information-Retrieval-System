from PyPDF2 import PdfReader
import os
import time
from utilities import utils
import jieba
import numpy as np


class Pypdf2Parser:
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.vocabulary = list()
        self.doc_ids = list()
        self.term_doc_incidence_matrix = None

    def parse_docs(self):
        start_time = time.time()
        self.doc_ids = utils.get_file_url_list(self.parent_dir)
        all_doc_term_list = list()

        for file_index in range(len(self.doc_ids)):
            reader = PdfReader(self.doc_ids[file_index])

            # Print the number of pages in the PDF
            utils.debug_print(f"There are {len(reader.pages)} Pages")

            curr_doc_term_list = list()
            # Go through every page and get the text
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_str = page.extract_text()
                # print(page_str)
                raw_seg_list = jieba.lcut(page_str)
                utils.debug_print('Start to process page ' + str(page_num+1) + ' out of ' + str(len(reader.pages)))
                utils.debug_print('Before: ' + str(len(raw_seg_list)))
                # print("Paddle Mode: " + ','.join(raw_seg_list))
                seg_list = list()
                for token_index in range(len(raw_seg_list)):
                    if utils.is_all_chinese(raw_seg_list[token_index]):
                        seg_list.append(raw_seg_list[token_index])
                utils.debug_print('After: ' + str(len(seg_list)))
                # print("Paddle Mode: " + ','.join(seg_list))
                self.vocabulary.extend(seg_list)
                self.vocabulary = list(set(self.vocabulary))
                utils.debug_print('Size of vocabulary: ' + str(len(self.vocabulary)))
                utils.debug_print(self.vocabulary)

                curr_doc_term_list.extend(seg_list)
            curr_doc_term_list = list(set(curr_doc_term_list))
            all_doc_term_list.append(curr_doc_term_list)

        shape = [len(self.doc_ids), len(self.vocabulary)]
        self.term_doc_incidence_matrix = np.zeros(shape, dtype=int)
        for doc_index in range(len(self.doc_ids)):
            for term_index in range(len(all_doc_term_list[doc_index])):
                curr_term = all_doc_term_list[doc_index][term_index]
                curr_term_index = self.vocabulary.index(curr_term)
                self.term_doc_incidence_matrix[doc_index, curr_term_index] = 1

        sparsity = np.count_nonzero(self.term_doc_incidence_matrix) / (shape[0]*shape[1])
        utils.debug_print('Sparsity of term_doc_incidence_matrix: ' + str(round(sparsity, 3)))

        end_time = time.time()
        utils.debug_print('Time for parse_docs: ' + str(end_time - start_time) + ' seconds')


if __name__ == '__main__':
    # 步骤一：解析原始文档
    start_time = time.time()
    pdf_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    my_pdf_parser = Pypdf2Parser(pdf_dir)
    my_pdf_parser.parse_docs()
    print('Time for parsing docs: ' + str(time.time() - start_time) + ' seconds')
