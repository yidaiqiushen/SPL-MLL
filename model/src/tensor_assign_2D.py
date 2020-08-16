# -*-coding:utf-8-*-
import tensorflow as tf


def tensor_expand(tensor_Input, Num):
    '''
    张量自我复制扩展，将Num个tensor_Input串联起来，生成新的张量，
    新的张量的shape=[tensor_Input.shape,Num]
    :param tensor_Input:
    :param Num:
    :return:
    '''
    tensor_Input = tf.expand_dims(tensor_Input, axis=0)
    tensor_Output = tensor_Input
    for i in range(Num - 1):
        tensor_Output = tf.concat([tensor_Output, tensor_Input], axis=0)
    return tensor_Output


def get_one_hot_matrix(height, width, position):
    '''
    生成一个 one_hot矩阵，shape=【height*width】，在position处的元素为1，其余元素为0
    :param height:
    :param width:
    :param position: 格式为【h_Index,w_Index】,h_Index,w_Index为int格式
    :return:
    '''
    col_length = height
    row_length = width
    col_one_position = position[0]
    row_one_position = position[1]
    rows_num = height
    cols_num = width

    single_row_one_hot = tf.one_hot(row_one_position, row_length, dtype=tf.float32)
    single_col_one_hot = tf.one_hot(col_one_position, col_length, dtype=tf.float32)

    one_hot_rows = tensor_expand(single_row_one_hot, rows_num)
    one_hot_cols = tensor_expand(single_col_one_hot, cols_num)
    one_hot_cols = tf.transpose(one_hot_cols)

    one_hot_matrx = one_hot_rows * one_hot_cols
    return one_hot_matrx


def tensor_assign_2D(tensor_input, position, value):
    '''
    给 2D tensor的特定位置元素赋值
    :param tensor_input: 输入的2D tensor，目前只支持2D
    :param position: 被赋值的张量元素的坐标位置，=【h_index,w_index】
    :param value:
    :return:
    '''
    shape = tensor_input.get_shape().as_list()
    height = shape[0]
    width = shape[1]
    h_index = position[0]
    w_index = position[1]
    one_hot_matrix = get_one_hot_matrix(height, width, position)

    new_tensor = tensor_input - tensor_input[h_index, w_index] * one_hot_matrix + one_hot_matrix * value

    return new_tensor