from typing import Callable

from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image


class UniversityDataset(Dataset):
    def __init__(
            self,
            root,
            drone_street_thumbnail_num: int = 16,
            transform_drone_street_thumbnail: Callable = None,
            transform_drone_street: Callable = None,
            transform_satellite: Callable = None,
            names=('satellite', 'street', 'drone', 'google')
    ):
        super(UniversityDataset).__init__()

        self.drone_street_thumbnail_num = drone_street_thumbnail_num

        self.transforms_satellite = transform_satellite
        self.transforms_drone_street = transform_drone_street
        self.transform_drone_street_thumbnail = transform_drone_street_thumbnail

        self.root = root
        self.names = names
        # 获取所有图片的相对路径分别放到对应的类别中
        # {satellite:{0839:[0839.jpg],0840:[0840.jpg]}}
        dict_path = {}
        for name in names:
            dict_ = {}
            for cls_name in os.listdir(os.path.join(root, name)):
                img_list = os.listdir(os.path.join(root, name, cls_name))
                img_path_list = [os.path.join(root, name, cls_name, img) for img in img_list]
                dict_[cls_name] = img_path_list
            dict_path[name] = dict_
            # dict_path[name+"/"+cls_name] = img_path_list

        # 获取设置名字与索引之间的镜像
        cls_names = os.listdir(os.path.join(root, names[0]))
        cls_names.sort()
        map_dict = {i: cls_names[i] for i in range(len(cls_names))}

        self.cls_names = cls_names
        self.map_dict = map_dict
        self.dict_path = dict_path
        self.index_cls_nums = 2

    # 从对应的类别中抽若干张出来
    def sample_from_cls(self, name, cls_num, sample_num: int = 1):
        img_paths = self.dict_path[name][cls_num]
        img_paths = np.random.choice(img_paths, sample_num)
        images = []
        for img_path in img_paths:
            img = Image.open(img_path)
            img = img.convert("RGB")
            images.append(img)
        return images

    def __getitem__(self, index):
        cls_nums = self.map_dict[index]
        image = self.sample_from_cls("satellite", cls_nums, 1)[0]
        img_sa = self.transforms_satellite(image)

        image = self.sample_from_cls("street", cls_nums, 1)[0]
        img_st = self.transforms_drone_street(image)

        image = self.sample_from_cls("drone", cls_nums, 1)[0]
        img_dr = self.transforms_drone_street(image)

        images = self.sample_from_cls("street", cls_nums, self.drone_street_thumbnail_num)
        thumb_st = [self.transform_drone_street_thumbnail(img) for img in images]

        images = self.sample_from_cls("drone", cls_nums, self.drone_street_thumbnail_num)
        thumb_dr = [self.transform_drone_street_thumbnail(img) for img in images]

        return img_sa, img_st, thumb_st, img_dr, thumb_dr, index

    def __len__(self):
        return len(self.cls_names)


class UniversitySampler(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source, batch_size=8, sample_num=4):
        self.data_len = len(data_source)
        self.batch_size = batch_size
        self.sample_num = sample_num

    def __iter__(self):
        lst = np.arange(0, self.data_len)
        np.random.shuffle(lst)
        nums = np.repeat(lst, self.sample_num, axis=0)
        return iter(nums)

    def __len__(self):
        return self.data_len
