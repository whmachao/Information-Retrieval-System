import jieba


class BasicQueryManager:
    def __init__(self, query_str):
        self.query_str = query_str
        self.query_term_list = None

    def parse_query(self):
        self.query_term_list = jieba.lcut(self.query_str)


if __name__ == '__main__':
    import os
    from doc_parsers import pdf_parser
    from index_managers.basic_inverted_indexer import BasicInvertedIndexer
    from utilities import utils

    # 步骤一：解析原始文档
    pdf_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    my_pdf_parser = pdf_parser.Pypdf2Parser(pdf_dir)
    my_pdf_parser.parse_docs()

    # 步骤二：构建索引
    my_doc_ids = my_pdf_parser.doc_ids
    my_vocabulary = my_pdf_parser.vocabulary
    my_term_doc_incidence_matrix = my_pdf_parser.term_doc_incidence_matrix
    my_basic_inverted_indexer = BasicInvertedIndexer(my_doc_ids, my_vocabulary, my_term_doc_incidence_matrix)
    my_basic_inverted_indexer.build_index()

    exit_flag = True
    while exit_flag:
        # 步骤三：用户输入查询并解析查询（自由文本，且其预处理逻辑应与原始文档解析中的逻辑保持一致）
        my_query_str = input('请输入查询：')
        if my_query_str == 'exit':
            exit_flag = False
            continue
        elif not utils.is_all_chinese(my_query_str):
            print('查询必须为中文，请重新输入！')
            continue
        my_basic_query_manager = BasicQueryManager(my_query_str)
        my_basic_query_manager.parse_query()
        print(my_basic_query_manager.query_term_list)