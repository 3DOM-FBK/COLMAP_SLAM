import copy
from pathlib import Path

def SortByTimeStamp(original_img_list):
    sorted_img_list = []
    imgs = copy.deepcopy(original_img_list)
    for i in range(len(imgs)):
        im = imgs[i]
        im = Path(im)
        name = str(Path(im.name).stem)
        sec = int(name[:10])
        nano_ = name[10:]
        missing = len(nano_)-9
        for m in range(missing):
            nano_ = nano_ + '0'
        nano = int(nano_)
        sorted_img_list.append((im, sec, nano))
    sorted_img_list.sort(key=lambda x: (x[1], x[2]))
    sorted_img_list = [t[0] for t in sorted_img_list]
    
    return sorted_img_list