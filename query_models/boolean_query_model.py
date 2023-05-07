import time


class BooleanQueryModel:
    def __init__(self, query_term_list, index):
        self.query_term_list = query_term_list
        self.index = index
        self.ranked_doc_ids = list()

    def execute_query(self):
        for term in self.query_term_list:
            if term in self.index.keys():
                term_inverted_index = self.index.get(term)
                term_doc_ids = term_inverted_index.get('doc_ids')
                if len(self.ranked_doc_ids) == 0:
                    self.ranked_doc_ids = term_doc_ids
                else:
                    self.ranked_doc_ids = self.ranked_doc_ids.intersection(term_doc_ids)


if __name__ == '__main__':
    import os
    from doc_parsers import pdf_parser
    from index_managers.basic_inverted_indexer import BasicInvertedIndexer
    from utilities import utils
    from query_managers.basic_query_manager import BasicQueryManager

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

        # 步骤四：依据用户查询在现有索引上进行基于向量空间模型的文档搜索及排序
        my_query_term_list = my_basic_query_manager.query_term_list
        my_index = my_basic_inverted_indexer.index
        my_boolean_query_model = BooleanQueryModel(my_query_term_list, my_index)
        my_boolean_query_model.execute_query()
        my_ranked_doc_ids = my_boolean_query_model.ranked_doc_ids
        if len(my_ranked_doc_ids) == 0:
            print('没有检索到任何相关文档，请重新输入查询！')
            continue
        else:
            print('相关文档如下所示：')
            for doc_index in range(len(my_ranked_doc_ids)):
                print(my_ranked_doc_ids[doc_index])

    print('本次信息检索系统设计与实现演示结束，祝大家学有所得！')