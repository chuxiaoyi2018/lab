import numpy as np


# 计算两个矩阵的余弦相似度
def cosine_similarity(matrix1, matrix2):
    # 确保两个矩阵形状相同
    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape."

    # 计算余弦相似度
    dot_product = np.dot(matrix1.flatten(), matrix2.flatten())
    norm_matrix1 = np.linalg.norm(matrix1)
    norm_matrix2 = np.linalg.norm(matrix2)
    similarity = dot_product / (norm_matrix1 * norm_matrix2)

    return similarity


# 计算两个矩阵差的绝对值的平均值
def mean_absolute_difference(matrix1, matrix2):
    # 确保两个矩阵形状相同
    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape."

    # 计算差的绝对值的平均值
    absolute_difference = np.abs(matrix1 - matrix2)
    mean_absolute_diff = np.mean(absolute_difference)

    return mean_absolute_diff


valid_length = 6
# valid_length = gt[name_list[0]].shape[1]
name_list = ["input_0", "input_1", "input_2"]
print("LAYER \tMAD_HS \tMAD_PK \tMAD_PV  \tCOS_HS \tCOS_PK \tCOS_PV")
for i in range(40):
    x = np.load(f"../python_demo/block_{i}.npz")
    gt = np.load(f"block_{i}.npz")
    # print(x.files)
    hidden_states = x[name_list[0]][:,:valid_length]
    # present_key = x[name_list[1]][:, : gt[name_list[1]].shape[2], :]
    present_key = x[name_list[1]][:,:valid_length]
    # present_key = np.transpose(present_key, (0, 2, 1, 3))
    present_value = x[name_list[2]][:,:valid_length]
    # present_value = x[name_list[2]][:, : gt[name_list[2]].shape[2], :]
    # present_value = np.transpose(present_value, (0, 2, 1, 3))
    print(
        "{:d}  \t{:.4f} \t{:.4f} \t{:.6f}  \t{:.4f} \t{:.4f} \t{:.6f}".format(
            i,
            mean_absolute_difference(hidden_states, gt[name_list[0]][:,:valid_length]),
            mean_absolute_difference(present_key, gt[name_list[1]][:,:valid_length]),
            mean_absolute_difference(present_value, gt[name_list[2]][:,:valid_length]),
            cosine_similarity(hidden_states, gt[name_list[0]][:,:valid_length]),
            cosine_similarity(present_key, gt[name_list[1]][:,:valid_length]),
            cosine_similarity(present_value, gt[name_list[2]][:,:valid_length]),
        )
    )
