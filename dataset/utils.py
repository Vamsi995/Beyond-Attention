import numpy as np
import pandas as pd
import torch

def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj


def generate_dataset(
    data, cov_data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    actual_data = data.copy()

    # speed_df, adj_matrix = load_sz_data()

    # # Convert data to PyTorch tensors
    # adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
    # speed_matrix = torch.tensor(speed_df.values, dtype=torch.float32)  # (2976, 156)

    # # Convert adjacency matrix to NetworkX graph
    # G = nx.from_numpy_array(adj_matrix.numpy(), create_using=nx.DiGraph)

    # # Compute Katz centrality
    # alpha = 0.01  # Damping factor
    # centrality_dict = nx.katz_centrality_numpy(G, alpha=alpha, beta=1.0)
    # centrality_values = torch.tensor([centrality_dict[i] for i in range(156)], dtype=torch.float32).reshape(1, -1)  # (1, 156)

    # # Expand centrality values to match time steps
    # centrality_matrix = centrality_values.repeat(speed_matrix.shape[0], 1)
    # data = np.expand_dims(data, axis=2)  # (2976, 156, 1)
    # centrality_expanded = np.expand_dims(centrality_matrix, axis=2)  # (2976, 156, 1)

    # Concatenate along the last axis
    # data = np.concatenate([data, centrality_expanded], axis=2)  # (2976, 156, 2)
    # print(data.shape)

    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    actual_train_data = data[:train_size]
    train_cov = cov_data[:train_size]
    test_data = data[train_size:time_len]
    actual_test_data = data[train_size:time_len]
    test_cov = cov_data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i : i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(
    data, cov_data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        cov_data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset


