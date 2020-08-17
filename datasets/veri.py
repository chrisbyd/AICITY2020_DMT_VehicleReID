import glob
import re
import os.path as osp
import xml.etree.ElementTree as ET

from .bases import BaseImageDataset


class VeRi(BaseImageDataset):
    """
       VeRi-776
       Reference:
       Liu, Xinchen, et al. "Large-scale vehicle re-identification in urban surveillance videos." ICME 2016.

       URL:https://vehiclereid.github.io/VeRi/

       Dataset statistics:
       # identities: 776
       # images: 37778 (train) + 1678 (query) + 11579 (gallery)
       # cameras: 20
       """

    dataset_dir = 'VeRi'

    def __init__(self, root='data', verbose=True, **kwargs):
        super(VeRi, self).__init__()
        self.root_dir = root
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.train_type_dir = osp.join(self.dataset_dir, "train_label.xml")
        self.test_type_dir = osp.join(self.dataset_dir, "test_label.xml")
        self.type_dict = dict()
        self.color_dict = dict()

        self._check_before_run()

        self._get_color_type_index()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)



        if verbose:
            print("=> VeRi-776 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.train_type_dir):
            raise RuntimeError("'{}' is not available".format(self.train_type_dir))


    def _get_color_type_index(self):
        parser = ET.XMLParser(encoding="utf-8")
        root = ET.parse(self.train_type_dir, parser=parser).getroot()
        for tag in root.findall("Items/Item"):
            imageName = tag.get("imageName")
            carID = imageName.split("_")[0]
            color = tag.get("colorID")
            car_type = tag.get("typeID")
            self.type_dict[carID] = int(car_type)
            self.color_dict[carID] = int(color)

        parser = ET.XMLParser(encoding="utf-8")
        root = ET.parse(self.test_type_dir, parser=parser).getroot()
        for tag in root.findall("Items/Item"):
            imageName = tag.get("imageName")
            color = tag.get("colorID")
            car_type = tag.get("typeID")
            self.type_dict[imageName] = int(car_type)
            self.color_dict[imageName] = int(color)



    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')
        unknown_train = {}
        unknown_test = {}

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 776  # pid == 0 means background
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            image_name = img_path.split("\\")[-1]
            carID = image_name.split("_")[0]
            if carID in self.color_dict and carID in self.type_dict:
                dataset.append((img_path, pid, camid, (self.type_dict[carID], self.color_dict[carID])))
            else:
                if "train" in img_path:
                    unknown_train[carID] = 1
                else:
                    unknown_test[carID] = 1

                dataset.append(
                    (img_path, pid, camid, (0, 0)))

        return dataset

