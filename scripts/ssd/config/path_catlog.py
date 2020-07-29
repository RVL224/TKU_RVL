import os


class DatasetCatalog:
    DATA_DIR = 'datasets'
    DATASETS = {
        ########################################################
        'voc_2007_trainval_ori': {
            "data_dir": "/workspace/datasets/VOCdevkit/VOC2007",
            "split": "trainval"
        },
        
        'voc_2012_trainval_ori': {
            "data_dir": "/workspace/datasets/VOCdevkit/VOC2012",
            "split": "trainval"
        },
        
        'voc_2007_test_ori': {
            # "data_dir": "/workspace/datasets/VOCdevkit/VOC2007",
            "data_dir": "/media/minda/storage/model_custom/dataset/voc/VOCdevkit/VOC2007",
            "split": "test"
        },
        ########################################################
        'voc_2007_train': {
            "data_dir": "/workspace/datasets/voc/VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "/workspace/datasets/voc/VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "/workspace/datasets/bdd100k/VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "/workspace/datasets/voc/VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "/workspace/datasets/voc/VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "/workspace/datasets/voc/VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "/workspace/datasets/voc/VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "/workspace/datasets/voc/VOC2012",
            "split": "test"
        },
        'voc_test': {
            "data_dir": "/workspace/datasets/bdd100k/VOC_test",
            "split": "test"
        },
        'voc_test1': {
            "data_dir": "/workspace/datasets/VOCdevkit/VOC2007",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "/workspace/datasets/coco/val2014",
            "ann_file": "/workspace/datasets/coco/annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "/workspace/datasets/coco/val2014",
            "ann_file": "/workspace/datasets/coco/annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "/workspace/datasets/coco/train2014",
            "ann_file": "/workspace/datasets/coco/annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "/workspace/datasets/coco/val2014",
            "ann_file": "/workspace/datasets/coco/annotations/instances_val2014.json"
        },
    }

    @staticmethod
    def get(name):
        if "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR
            if 'VOC_ROOT' in os.environ:
                voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)
        elif "coco" in name:
            coco_root = DatasetCatalog.DATA_DIR
            if 'COCO_ROOT' in os.environ:
                coco_root = os.environ['COCO_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(coco_root, attrs["data_dir"]),
                ann_file=os.path.join(coco_root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
