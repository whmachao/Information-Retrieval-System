import os
from os import listdir
from os.path import join, isfile, isdir


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


if __name__ == '__main__':
    my_parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    my_file_url_list = get_file_url_list(my_parent_dir)
    print(my_parent_dir)
    print(my_file_url_list)
