import time
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
from utilities.utils import cosine_similarity
from utilities import utils


class VectorQueryModel:
    def __init__(self, query_term_list, index):
        self.query_term_list = query_term_list
        self.index = index
        self.ranked_doc_ids = list()
        print('您正在使用向量空间模型进行检索')

    def execute_query(self):
        start_time = time.time()
        # 计算查询向量
        query_vector = list()
        for key in self.index.keys():
            if key in self.query_term_list:
                query_vector.append(1)
            else:
                query_vector.append(0)
        query_vector = np.array(query_vector)

        # 计算所有的文档向量
        doc_ids_list = list()
        for key in self.index.keys():
            doc_ids_list.extend(self.index.get(key).get('doc_ids'))
        unique_doc_ids_list = list(set(doc_ids_list))
        doc_num = len(unique_doc_ids_list)
        all_doc_vectors = np.zeros(shape=(doc_num, len(self.index.keys())))
        for doc_index in range(doc_num):
            term_index = 0
            for key in self.index.keys():
                if unique_doc_ids_list[doc_index] in self.index.get(key).get('doc_ids'):
                    all_doc_vectors[doc_index, term_index] = 1
                term_index += 1

        # 计算查询向量与所有文档向量的相似度
        similarity_list = list()
        for row_index in range(all_doc_vectors.shape[0]):
            curr_doc_vec = all_doc_vectors[row_index]
            curr_similarity = cosine_similarity(query_vector, curr_doc_vec)
            # 若用户查询向量为0，则结束查询过程
            if curr_similarity is None:
                return
            similarity_list.append(curr_similarity)
        # 依据查询向量与所有文档向量的相似度从大到小对所有文档进行排序
        for similarity_index in range(len(unique_doc_ids_list)):
            maximum_similarity = max(similarity_list)
            top_ranked_doc_index = similarity_list.index(maximum_similarity)
            top_ranked_doc_id = unique_doc_ids_list[top_ranked_doc_index]
            self.ranked_doc_ids.append(top_ranked_doc_id)
            similarity_list[top_ranked_doc_index] = -0.01
        end_time = time.time()
        utils.debug_print('Time for execute_query: ' + str(end_time - start_time) + ' seconds')


if __name__ == '__main__':
    import os
    from doc_parsers.pdf_parser import Pypdf2Parser
    from index_managers.basic_inverted_indexer import BasicInvertedIndexer
    from query_managers.basic_query_manager import BasicQueryManager

    # 步骤一：解析原始文档
    my_start_time = time.time()
    pdf_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    my_pdf_parser = Pypdf2Parser(pdf_dir)
    my_pdf_parser.parse_docs()
    print('Time for parsing docs: ' + str(time.time()-my_start_time) + ' seconds')

    # 步骤二：构建索引
    my_start_time = time.time()
    my_doc_ids = my_pdf_parser.doc_ids
    my_vocabulary = my_pdf_parser.vocabulary
    my_term_doc_incidence_matrix = my_pdf_parser.term_doc_incidence_matrix
    my_basic_inverted_indexer = BasicInvertedIndexer(my_doc_ids, my_vocabulary, my_term_doc_incidence_matrix)
    my_basic_inverted_indexer.build_index()
    print('Time for building index: ' + str(time.time() - my_start_time) + ' seconds')

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
        my_start_time = time.time()
        my_query_term_list = my_basic_query_manager.query_term_list
        my_index = my_basic_inverted_indexer.index
        my_boolean_query_model = VectorQueryModel(my_query_term_list, my_index)
        my_boolean_query_model.execute_query()
        my_ranked_doc_ids = my_boolean_query_model.ranked_doc_ids
        if len(my_ranked_doc_ids) == 0:
            print('没有检索到任何相关文档，请重新输入查询！')
            continue
        else:
            print('相关文档如下所示：')
            for my_doc_index in range(len(my_ranked_doc_ids)):
                print(my_ranked_doc_ids[my_doc_index])
        print('Time for executing query: ' + str(time.time() - my_start_time) + ' seconds')

    print('本次信息检索系统设计与实现演示结束，祝大家学有所得！')
