import os.path as osp
import os
import shutil
import numpy as np
import torch
import cv2

def mkdirs(d, replace = True):
    if not osp.exists(d):
        os.makedirs(d)
    elif replace == True:
        shutil.rmtree(d)
        os.makedirs(d)
        

def get_boxes(boxes, target_x, target_y, target_w, target_h, exclude_indices = []):
    x = boxes[:,0] + (boxes[:,2] - boxes[:,0])/2
    y = boxes[:,1] + (boxes[:,3] - boxes[:,1])/2
    w = boxes[:,2] - boxes[:,0]
    h = boxes[:,3] - boxes[:,1]

    # Calculate absolute differences
    diff_x = torch.abs(x - target_x)
    diff_y = torch.abs(y - target_y)
    diff_w = torch.abs(w - target_w)
    diff_h = torch.abs(h - target_h)

    # Sum the differences for each box
    total_diff = diff_x + diff_y + diff_w + diff_h

    # Set total_diff at excluded indices to a large value (infinity)
    total_diff[exclude_indices] = float('inf')

    # Find the index with the smallest difference, excluding specified indices
    closest_idx = torch.argmin(total_diff)

    return closest_idx


# import os
# import numpy as np

def preprocessing_DanceTrack1(seq_root, trainer, detector_model):
    for type in trainer:
        label_root = os.path.join(seq_root, type, 'trackers_gt_t_1')
        mkdirs(label_root)

        seq_root_tr = os.path.join(seq_root, type)
        
        # Filter out hidden files like .DS_Store
        seqs = [s for s in os.listdir(seq_root_tr) if not s.startswith('.') and "gt_t" not in s]

        for seq in seqs:
            print(seq)
            seq_info_path = os.path.join(seq_root_tr, seq, 'seqinfo.ini')

            if not os.path.exists(seq_info_path):
                print(f"Warning: {seq_info_path} not found, skipping sequence '{seq}'.")
                continue
            
            with open(seq_info_path) as f:
                seq_info = f.read()

            # Extract dimensions (only do this once per sequence)
            seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
            seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

            gt_txt = os.path.join(seq_root_tr, seq, 'gt', 'gt.txt')

            if not os.path.exists(gt_txt):
                print(f"Warning: {gt_txt} not found for sequence '{seq}', skipping.")
                continue

            gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
            idx = np.lexsort(gt.T[:2, :])  # sort by fid, tid
            gt = gt[idx, :]

            seq_img_path = {}
            seq_label_root = os.path.join(label_root, seq, 'img1')
            mkdirs(seq_label_root)

            for fid, tid, x, y, w, h, mark, cls, vis in gt:
                frame_path = os.path.join(seq_root_tr, seq, 'img1', f"{int(fid):08d}.jpg")
                if not os.path.exists(frame_path):
                    print(f"Warning: {frame_path} does not exist.")
                    continue;  # Skip this frame if the file doesn't exist
                if frame_path not in seq_img_path:
                    result = detector_model.get_object(frame_path)
                    seq_img_path[frame_path] = result
                else:
                    result = seq_img_path[frame_path]

                obj_feats = result.feats
                boxes = result.boxes.data
                exclude_indices = [i for i in range(len(boxes)) if boxes[i][-1] != 0]

                if mark == 0 or cls != 1:
                    continue
                
                x += w / 2
                y += h / 2

                indicase_box = get_boxes(boxes, target_x=x, target_y=y, target_w=w, target_h=h, exclude_indices=exclude_indices)
                x_norm, y_norm, w_norm, h_norm = x / seq_width, y / seq_height, w / seq_width, h / seq_height

                # Feature extraction
                feature = obj_feats[indicase_box].tolist()

                label_fpath = os.path.join(seq_label_root, f"{int(tid):06d}.txt")

                # Create label string
                label_str = f"{0} {int(fid)} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f} {int(vis)} {seq_height} {seq_width}"
                label_str += ' ' + ' '.join([f'{feat:.6f}' for feat in feature]) + '\n'

                # print(label_str)

                with open(label_fpath, 'a') as f:
                    f.write(label_str)

def preprocessing_DanceTrack(seq_root, trainer):
    for type in trainer:
        label_root = os.path.join(seq_root, type, 'trackers_gt_t')
        mkdirs(label_root)

        seq_root_tr = os.path.join(seq_root, type)
        
        # Filter out hidden files like .DS_Store
        seqs = [s for s in os.listdir(seq_root_tr) if not s.startswith('.') and "gt_t" not in s]

        for seq in seqs:
            print(seq)
            seq_info_path = os.path.join(seq_root_tr, seq, 'seqinfo.ini')

            if not os.path.exists(seq_info_path):
                print(f"Warning: {seq_info_path} not found, skipping sequence '{seq}'.")
                continue
            
            with open(seq_info_path) as f:
                seq_info = f.read()

            # Extract dimensions (only do this once per sequence)
            seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
            seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

            gt_txt = os.path.join(seq_root_tr, seq, 'gt', 'gt.txt')

            if not os.path.exists(gt_txt):
                print(f"Warning: {gt_txt} not found for sequence '{seq}', skipping.")
                continue

            gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
            idx = np.lexsort(gt.T[:2, :])  # sort by fid, tid
            gt = gt[idx, :]

            seq_label_root = os.path.join(label_root, seq, 'img1')
            mkdirs(seq_label_root, replace= False)

            for fid, tid, x, y, w, h, mark, cls, vis in gt:
                frame_path = os.path.join(seq_root_tr, seq, 'img1', f"{int(fid):08d}.jpg")
                if not os.path.exists(frame_path):
                    print(f"Warning: {frame_path} does not exist.")
                    continue;  # Skip this frame if the file doesn't exist
                
                

                if mark == 0 or cls != 1:
                    continue
                
                x += w / 2
                y += h / 2
                x_norm, y_norm, w_norm, h_norm = x / seq_width, y / seq_height, w / seq_width, h / seq_height

                label_fpath = os.path.join(seq_label_root, f"{int(tid):06d}.txt")

                # Create label string
                label_str = f"{0} {int(fid)} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f} {int(vis)} {seq_height} {seq_width}"

                # print(label_str)

                with open(label_fpath, 'a') as f:
                    f.write(label_str)



def preprocessing_crop_Obj(seq_root, trainer = ['train', "val", 'test']):

    for type in trainer:
        label_root = os.path.join(seq_root, 'reid_gt_t', type)
        mkdirs(label_root, replace = False)

        seq_root_tr = os.path.join(seq_root, type)
        
        seqs = [s for s in os.listdir(seq_root_tr) if not s.startswith('.') and "gt_t" not in s]

        for seq in seqs:
            print(seq)
            seq_info_path = os.path.join(seq_root_tr, seq, 'seqinfo.ini')

            if not os.path.exists(seq_info_path):
                print(f"Warning: {seq_info_path} not found, skipping sequence '{seq}'.")
                continue
            
            with open(seq_info_path) as f:
                seq_info = f.read()

            gt_txt = os.path.join(seq_root_tr, seq, 'gt', 'gt.txt')

            if not os.path.exists(gt_txt):
                print(f"Warning: {gt_txt} not found for sequence '{seq}', skipping.")
                continue

            gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
            idx = np.lexsort(gt.T[:2, :])  # sort by fid, tid
            gt = gt[idx, :]

            seq_label_root = os.path.join(label_root, seq, 'img1')
            mkdirs(seq_label_root, replace = False)

            for fid, tid, x, y, w, h, mark, cls, vis in gt:
                frame_path = os.path.join(seq_root_tr, seq, 'img1', f"{int(fid):08d}.jpg")
                if not os.path.exists(frame_path):
                    print(f"Warning: {frame_path} does not exist.")
                    continue

                if mark == 0 or cls != 1:
                    continue

                org_img = cv2.imread(frame_path)
                if org_img is None:
                    print(f"Warning: Could not read image from {frame_path}")
                    continue

                # Define label file path and ensure the directory exists
                label_fpath = os.path.join(seq_label_root, f"{int(tid):06d}/{int(fid):06d}.jpg")
                os.makedirs(os.path.dirname(label_fpath), exist_ok=True)

                # Crop the image and write to file
                cv2.imwrite(label_fpath, org_img[int(y):int(y+h)+1, int(x):int(x+w)+1])




# def preprocessing_DanceTrack(seq_root, trainer):

#     for type in trainer:
#         # if type == "test":
#         #     continue;
#         label_root = osp.join(seq_root, type, 'trackers_gt_t')
#         mkdirs(label_root)

#         seq_root_tr = osp.join(seq_root, type)
        
#         # Filter out hidden files like .DS_Store
#         seqs = [s for s in os.listdir(seq_root_tr) if not s.startswith('.')  and "gt_t" not in s]
        
#         for seq in seqs:
#             print(seq)
#             # Construct path to seqinfo.ini file
#             seq_info_path = osp.join(seq_root_tr, seq, 'seqinfo.ini')
            
#             # Check if seqinfo.ini exists
#             if not osp.exists(seq_info_path):
#                 print(f"Warning: {seq_info_path} not found, skipping sequence '{seq}'.")
#                 continue
            
#             with open(seq_info_path) as f:
#                 seq_info = f.read()

#             # Extract dimensions from seqinfo
#             seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
#             seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

#             gt_txt = osp.join(seq_root_tr, seq, 'gt', 'gt.txt')
            
#             # Check if gt.txt file exists
#             if not osp.exists(gt_txt):
#                 print(f"Warning: {gt_txt} not found for sequence '{seq}', skipping.")
#                 continue

#             gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
#             idx = np.lexsort(gt.T[:2, :])
#             gt = gt[idx, :]

#             seq_label_root = osp.join(label_root, seq, 'img1')
#             mkdirs(seq_label_root)

#             for fid, tid, x, y, w, h, mark, cls, vis in gt:
#                 if mark == 0 or not cls == 1:
#                     continue
#                 fid = int(fid)
#                 tid = int(tid)
#                 x += w / 2
#                 y += h / 2
#                 label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(tid))
#                 label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
#                     fid, x / seq_width, y / seq_height, w / seq_width, h / seq_height, vis)
#                 with open(label_fpath, 'a') as f:
#                     f.write(label_str)

