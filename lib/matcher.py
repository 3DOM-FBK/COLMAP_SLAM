import torch
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from kornia import feature

def Matcher(desc_1, desc_2, kornia_matcher : str, ratio_threshold):
    torch_desc_1 = torch.from_numpy(desc_1)
    torch_desc_2 = torch.from_numpy(desc_2)
    matcher = feature.DescriptorMatcher(match_mode=kornia_matcher, th=ratio_threshold)
    match_distances, matches_matrix = matcher.forward(torch_desc_1, torch_desc_2)

    return matches_matrix


def UpdateAdjacencyMatrix(adjacency_matrix : np.ndarray[bool], kfm_batch : list, overlap : int, first_loop: bool) -> np.ndarray[bool]:
    if first_loop == True:
        matrix_dim = len(kfm_batch)
        adjacency_matrix = np.full((matrix_dim, matrix_dim), False, dtype=bool)

        # Define matches
        for i in range(1, matrix_dim):
            j = i-overlap
            for k in range(j,i):
                if k >= 0:
                    adjacency_matrix[i, k] = True
                    adjacency_matrix[k, i] = True
    
    else:
        old_matrix_shape = adjacency_matrix.shape
        new_matrix_shape = (len(kfm_batch), len(kfm_batch))
        row_pad = new_matrix_shape[0] - old_matrix_shape[0]
        col_pad = new_matrix_shape[1] - old_matrix_shape[1]

        # Pad the input matrix with False values
        adjacency_matrix = np.pad(adjacency_matrix, ((0, row_pad), (0, col_pad)), constant_values=False)

        # Define matches
        for i in range(old_matrix_shape[0]-1, new_matrix_shape[0]):
            j = i-overlap
            for k in range(j,i):
                if k >= 0:
                    adjacency_matrix[i, k] = True
                    adjacency_matrix[k, i] = True

    return adjacency_matrix

def PlotAdjacencyMatrix(adjacency_matrix : np.ndarray[bool]) -> None:
    # Create a colormap where True values are blue and False values are red
    cmap = plt.cm.get_cmap('RdYlBu')
    cmap.set_bad(color='red')
    cmap.set_over(color='blue')

    # Plot the input matrix with color mapping
    plt.imshow(adjacency_matrix, cmap=cmap, vmin=False, vmax=True)
    plt.colorbar()

    # Display the plot
    plt.show()


if __name__ == "__main__":

    print('Test UpdateAdjacencyMatrix()')

    adjacency_matrix = None
    kfm_batch = [1, 2, 3, 6, 7, 8, 9, 10]
    overlap = 2

    adjacency_matrix = UpdateAdjacencyMatrix(adjacency_matrix, kfm_batch, overlap, first_loop=True)

    kfm_batch = [1, 2, 3, 6, 7, 8, 9, 10, 12, 17, 20, 21]
    overlap = 5
    adjacency_matrix = UpdateAdjacencyMatrix(adjacency_matrix, kfm_batch, overlap, first_loop=False)

    PlotAdjacencyMatrix(adjacency_matrix)

