import cv2
import h5py
import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from kornia import feature

from pathlib import Path
from typing import List, Tuple, Union

from lib.thirdparty.SuperGlue.models.utils import read_image
from lib.thirdparty.LightGlue.lightglue import LightGlue, SuperPoint, DISK
from lib.thirdparty.LightGlue.lightglue.utils import load_image, rbd
from lib import h5_to_db, db_colmap
from copy import deepcopy
from collections import defaultdict

def load_torch_image(fname, device=torch.device('cpu')):
    img = kornia.image_to_tensor(cv2.imread(fname), False).float() /255.
    img = kornia.color.bgr_to_rgb(img.to(device))
    return img

def Matcher(desc_1, desc_2, kornia_matcher : str, ratio_threshold, kps1, kps2, laf1, laf2):
    torch_desc_1 = torch.from_numpy(desc_1)
    torch_desc_2 = torch.from_numpy(desc_2)

    if kornia_matcher == 'adalam':
        # See: 
        # https://github.com/ducha-aiki/imc2023-kornia-starter-pack/blob/main/DISK-adalam-pycolmap-3dreconstruction.ipynb
        # https://kornia.readthedocs.io/en/latest/feature.html#local-affine-frames-laf
        # https://github.com/ducha-aiki/imc2023-kornia-starter-pack/blob/main/DISK-adalam-pycolmap-3dreconstruction.ipynb
        
        #laf1 = torch.from_numpy(np.zeros((1,desc_1.shape[0],2,3)))
        #laf2 = torch.from_numpy(np.zeros((1,desc_2.shape[0],2,3)))

        kps1_tensor = torch.from_numpy(kps1[:,:2]).to(device='cuda')
        if laf1.any == None and laf2.any == None:
            laf1 = feature.laf_from_center_scale_ori(kps1_tensor.unsqueeze(0),
                                                 torch.ones(1, len(kps1), 1, 1,device='cuda'))
        else:
            laf1 = torch.from_numpy(laf1)
        kps2_tensor = torch.from_numpy(kps2[:,:2]).to(device='cuda')

        if laf1.any == None and laf2.any == None:
            laf2 = feature.laf_from_center_scale_ori(kps2_tensor.unsqueeze(0),
                                                 torch.ones(1, len(kps2), 1, 1,device='cuda'))
        else:
            laf2 = torch.from_numpy(laf2)

        match_distances, matches_matrix = feature.match_adalam(torch_desc_1, torch_desc_2, laf1, laf2)


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
        # Could be useful implement the use of LAF (Local Affine Frames)
        # See https://kornia.readthedocs.io/en/latest/feature.html#local-affine-frames-laf
        # See https://github.com/ducha-aiki/imc2023-kornia-starter-pack/blob/main/DISK-adalam-pycolmap-3dreconstruction.ipynb
        # kpts1 = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
        # So the ktps position should be updated

    elif kornia_matcher =='nn' or kornia_matcher == 'snn' or kornia_matcher == 'mnn' or kornia_matcher == 'smnn':
        matcher = feature.DescriptorMatcher(match_mode=kornia_matcher, th=ratio_threshold)
        match_distances, matches_matrix = matcher.forward(torch_desc_1, torch_desc_2)
    else:
        print('Insert a matcher between those available in conf.ini\n Exit')
        quit()

    return matches_matrix, kps1, kps2


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

def get_unique_idxs(A, dim=1):
    # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0],device=cum_sum.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    return first_indicies

def LoFTR(keyframe_dir : List[Path], 
          images_dict, 
          ij, 
          path_to_database,
          min_matches=15, 
          resize_to_ = (640, 480)):
    device=torch.device('cuda')
    matcher = feature.LoFTR(pretrained='indoor').to(device).eval()

    loftr_keypoints = {}
    loftr_matches = {}

    with h5py.File(f'./matches_loftr.h5', mode='w') as f_match:
        for i, j in ij:
            fname1, fname2 = keyframe_dir / images_dict[j + 1], keyframe_dir / images_dict[i + 1]
            # Load img1
            timg1 = kornia.color.rgb_to_grayscale(load_torch_image(str(fname1), device=device))
            H1, W1 = timg1.shape[2:]
            if H1 < W1:
                resize_to = resize_to_[1], resize_to_[0]
            else:
                resize_to = resize_to_
            timg_resized1 = kornia.geometry.resize(timg1, resize_to, antialias=True)
            h1, w1 = timg_resized1.shape[2:]
            # Load img2
            timg2 = kornia.color.rgb_to_grayscale(load_torch_image(str(fname2), device=device))
            H2, W2 = timg2.shape[2:]
            if H2 < W2:
                resize_to2 = resize_to[1], resize_to[0]
            else:
                resize_to2 = resize_to_
            timg_resized2 = kornia.geometry.resize(timg2, resize_to2, antialias=True)
            h2, w2 = timg_resized2.shape[2:]
            with torch.inference_mode():
                input_dict = {"image0": timg_resized1,"image1": timg_resized2}
                correspondences = matcher(input_dict)
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            mkpts0[:,0] *= float(W1) / float(w1)
            mkpts0[:,1] *= float(H1) / float(h1)
            mkpts1[:,0] *= float(W2) / float(w2)
            mkpts1[:,1] *= float(H2) / float(h2)
            n_matches = len(mkpts1)
            group  = f_match.require_group(str(j+1))
            if n_matches >= min_matches:
                 group.create_dataset(str(i+1), data=np.concatenate([mkpts0, mkpts1], axis=1))
    # Let's find unique loftr pixels and group them together
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts=defaultdict(int)
    with h5py.File(f'./matches_loftr.h5', mode='r') as f_match:
        for k1 in f_match.keys():
            group  = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1]
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0]+=total_kpts[k1]
                current_match[:, 1]+=total_kpts[k2]
                total_kpts[k1]+=len(matches)
                total_kpts[k2]+=len(matches)
                match_indexes[k1][k2]=current_match
    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]),dim=0, return_inverse=True)
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:,0] = unique_match_idxs[k1][m2[:,0]]
            m2[:,1] = unique_match_idxs[k2][m2[:,1]]
            mkpts = np.concatenate([unique_kpts[k1][ m2[:,0]],
                                    unique_kpts[k2][  m2[:,1]],
                                   ],
                                   axis=1)
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()


    db = db_colmap.COLMAPDatabase.connect(path_to_database)
    with h5py.File(f'./keypoints.h5', mode='w') as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1
            db.add_keypoints(k, kpts1)
    
    with h5py.File(f'./matches.h5', mode='w') as f_match:
        for k1, gr in out_match.items():
            group  = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match
                db.add_two_view_geometry(int(k1), int(k2), match)
    
    db.commit()
    db.close()

    return

def UpdateAdjacencyMatrix(adjacency_matrix : np.ndarray[bool], kfm_batch : list, overlap : int, first_loop: bool, n_cameras: int) -> np.ndarray[bool]:
    if first_loop == True:
        matrix_dim = int(len(kfm_batch)/n_cameras)      
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
        new_matrix_shape = (int(len(kfm_batch)/n_cameras), int(len(kfm_batch)/n_cameras))
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

