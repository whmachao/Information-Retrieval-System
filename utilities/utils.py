import os
from os import listdir
from os.path import join, isfile, isdir
import numpy as np
from utilities import Constants


def get_file_url_list(parent_dir, file_type='pdf', file_url_list=list()):
    for f in listdir(parent_dir):
        temp_dir = join(parent_dir, f)
        if isfile(temp_dir) and temp_dir.endswith(file_type):
            file_url_list.append(temp_dir)
        elif isdir(temp_dir):
            get_file_url_list(temp_dir, file_type, file_url_list)
    return file_url_list


def is_all_chinese(token_str):
    for _char in token_str:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def cosine_similarity(vec_x, vec_y):
    num = vec_x.dot(vec_y.T)
    if num == 0.0:
        return None
    denom = np.linalg.norm(vec_x) * np.linalg.norm(vec_y)
    similarity = num / denom
    return similarity


def debug_print(debug_info, mode=Constants.DEBUG_MODE):
    if mode:
        print(debug_info)


def create_query_model(query_model_name, query_term_list, index):
    if query_model_name == Constants.QUERY_MODEL_NAMES[0]:
        from query_models.boolean_query_model import BooleanQueryModel
        return BooleanQueryModel(query_term_list, index)
    if query_model_name == Constants.QUERY_MODEL_NAMES[1]:
        from query_models.vector_query_model import VectorQueryModel
        return VectorQueryModel(query_term_list, index)

    raise ValueError(query_model_name + ' is not a supported query model!')


if __name__ == '__main__':
    my_parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    my_file_url_list = get_file_url_list(my_parent_dir)
    print(my_parent_dir)
    print(my_file_url_list)
