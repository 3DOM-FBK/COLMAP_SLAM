import torch
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from kornia import feature

from pathlib import Path
from typing import List, Tuple, Union

from lib.thirdparty.SuperGlue.models.utils import read_image
from lib.thirdparty.LightGlue.lightglue import LightGlue, SuperPoint, DISK
from lib.thirdparty.LightGlue.lightglue.utils import load_image, rbd

def Matcher(desc_1, desc_2, kornia_matcher : str, ratio_threshold, kps1, kps2):
    torch_desc_1 = torch.from_numpy(desc_1)
    torch_desc_2 = torch.from_numpy(desc_2)

    if kornia_matcher == 'adalam':
        # For adalam see: https://github.com/ducha-aiki/imc2023-kornia-starter-pack/blob/main/DISK-adalam-pycolmap-3dreconstruction.ipynb
        #laf1 = torch.from_numpy(np.zeros((1,desc_1.shape[0],2,3)))
        #laf2 = torch.from_numpy(np.zeros((1,desc_2.shape[0],2,3)))
        kps1_tensor = torch.from_numpy(kps1[:,:2]).to(device='cuda')
        laf1 = feature.laf_from_center_scale_ori(kps1_tensor.unsqueeze(0),
                                             torch.ones(1, len(kps1), 1, 1,device='cuda'))
        kps2_tensor = torch.from_numpy(kps2[:,:2]).to(device='cuda')
        laf2 = feature.laf_from_center_scale_ori(kps2_tensor.unsqueeze(0),
                                             torch.ones(1, len(kps2), 1, 1,device='cuda'))
        match_distances, matches_matrix = feature.match_adalam(torch_desc_1, torch_desc_2, laf1, laf2)
        # Should be used refined kpts
        # kpts1 = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
        # See the kornia tutorial

    elif kornia_matcher == 'lightglue':
        #extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
        #image0 = load_image(r'C:\Users\lmorelli\Desktop\Luca\GitHub_3DOM\COLMAP_SLAMaaa\COLMAP_SLAM\imgs\1403636753563555584.jpg').cuda()
        #feats0 = extractor.extract(image0)
        #print(feats0)
        #quit()
        #print(torch_desc_1.unsqueeze(0))
        #print(torch_desc_1.shape)
        #quit()
        #matcher = LightGlue(features='disk').eval().cuda()
        #matches01 = matcher({'image0': {"keypoints" : torch_desc_1.unsqueeze(0), 'image_size' : torch.from_numpy(np.array([[752, 480]]))}, 'image1': {"keypoints" : torch_desc_2.unsqueeze(0), 'image_size' : torch.from_numpy(np.array([[752, 480]]))}})
        #print(matches01)
        #laf1 = np.zeros((1,desc_1.shape[0],2,3))
        #laf2 = np.zeros((1,desc_2.shape[0],2,3))
        kps1_tensor = torch.from_numpy(kps1[:,:2]).to(device='cuda')
        laf1 = feature.laf_from_center_scale_ori(kps1_tensor.unsqueeze(0),
                                             torch.ones(1, len(kps1), 1, 1,device='cuda'))
        kps2_tensor = torch.from_numpy(kps2[:,:2]).to(device='cuda')
        laf2 = feature.laf_from_center_scale_ori(kps2_tensor.unsqueeze(0),
                                             torch.ones(1, len(kps2), 1, 1,device='cuda'))
        LGlue = feature.LightGlueMatcher(feature_name='superpoint')
        desc1_torch = torch.from_numpy(desc_1/np.max(desc_1)).to(dtype=torch.float32)
        desc2_torch = torch.from_numpy(desc_2/np.max(desc_2)).to(dtype=torch.float32)
        match_distances, matches_matrix = LGlue.forward(desc1_torch.cpu(), desc2_torch.cpu(), laf1.cpu(), laf2.cpu())
        matches_matrix = matches_matrix.cpu().detach().numpy()
        #print(matches_matrix)

    elif kornia_matcher =='nn' or kornia_matcher == 'snn' or kornia_matcher == 'mnn' or kornia_matcher == 'smnn':
        matcher = feature.DescriptorMatcher(match_mode=kornia_matcher, th=ratio_threshold)
        match_distances, matches_matrix = matcher.forward(torch_desc_1, torch_desc_2)
    else:
        print('Insert a matcher between those available in conf.ini\n Exit')
        quit()

    return matches_matrix

def SuperGlue(keyframe_dir : List[Path], im1 : str, im2 : str, matching):

    # Load the image pair.
    image0, inp0, scales0 = read_image(
        keyframe_dir / im1, "cuda", [-1], 0, False)
    image1, inp1, scales1 = read_image(
        keyframe_dir / im2, "cuda", [-1], 0, False)

    # Perform the matching.
    pred = matching({'image0': inp0, 'image1': inp1})
    #pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    return kpts0, kpts1, matches


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

