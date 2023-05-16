import time
import os
from doc_parsers.pdf_parser import Pypdf2Parser
from index_managers.basic_inverted_indexer import BasicInvertedIndexer
from utilities import utils
from query_managers.basic_query_manager import BasicQueryManager
from utilities import Constants


if __name__ == '__main__':
    # 步骤一：解析原始文档
    start_time = time.time()
    doc_dir = os.path.abspath(os.path.join(os.getcwd(), "../Information-Retrieval-System/archives/IR-Course-Reports"))
    my_doc_parser = Pypdf2Parser(doc_dir)
    my_doc_parser.parse_docs()
    print('Time for parsing docs: ' + str(time.time()-start_time) + ' seconds')

    # 步骤二：构建索引
    start_time = time.time()
    my_doc_ids = my_doc_parser.doc_ids
    my_vocabulary = my_doc_parser.vocabulary
    my_term_doc_incidence_matrix = my_doc_parser.term_doc_incidence_matrix
    my_basic_inverted_indexer = BasicInvertedIndexer(my_doc_ids, my_vocabulary, my_term_doc_incidence_matrix)
    my_basic_inverted_indexer.build_index()
    print('Time for building index: ' + str(time.time() - start_time) + ' seconds')

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
        start_time = time.time()
        my_query_term_list = my_basic_query_manager.query_term_list
        my_index = my_basic_inverted_indexer.index
        my_query_model_name = Constants.QUERY_MODEL_NAMES[0]
        my_boolean_query_model = utils.create_query_model(my_query_model_name, my_query_term_list, my_index)
        my_boolean_query_model.execute_query()
        my_ranked_doc_ids = my_boolean_query_model.ranked_doc_ids
        if len(my_ranked_doc_ids) == 0:
            print('没有检索到任何相关文档，请重新输入查询！')
            continue
        else:
            print('相关文档如下所示：')
            for my_doc_index in range(len(my_ranked_doc_ids)):
                print(my_ranked_doc_ids[my_doc_index])
        print('Time for executing query: ' + str(time.time() - start_time) + ' seconds')

    print('本次信息检索系统设计与实现演示结束，祝大家学有所得！')
