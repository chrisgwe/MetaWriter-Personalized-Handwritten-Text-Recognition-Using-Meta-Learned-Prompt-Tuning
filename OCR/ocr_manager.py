from basic.generic_training_manager import GenericTrainingManager
import os
from PIL import Image
import pickle


class OCRManager(GenericTrainingManager):
    def __init__(self, params):
        super(OCRManager, self).__init__(params)
        if self.dataset is not None:
            self.params["model_params"]["vocab_size"] = len(self.dataset.charset)

    def generate_syn_line_dataset(self, name):
        """
        Generate synthetic line dataset from currently loaded dataset
        """
        dataset_name = list(self.params['dataset_params']["datasets"].keys())[0]
        path = os.path.join(os.path.dirname(self.params['dataset_params']["datasets"][dataset_name]), name)
        os.makedirs(path, exist_ok=True)
        charset = set()
        dataset = None
        gt = {
            "train": dict(),
            "valid": dict(),
            "test": dict()
        }
        for set_name in ["train", "valid", "test"]:
            set_path = os.path.join(path, set_name)
            os.makedirs(set_path, exist_ok=True)
            if set_name == "train":
                dataset = self.dataset.train_dataset
            elif set_name == "valid":
                dataset = self.dataset.valid_datasets["{}-valid".format(dataset_name)]
            elif set_name == "test":
                self.dataset.generate_test_loader("{}-test".format(dataset_name), [(dataset_name, "test"), ])
                dataset = self.dataset.test_datasets["{}-test".format(dataset_name)]

            samples = list()
            for sample in dataset.samples:
                for line_label in sample["label"].split("\n"):
                    for chunk in [line_label[i:i+100] for i in range(0, len(line_label), 100)]:
                        charset = charset.union(set(chunk))
                        if len(chunk) > 0:
                            samples.append({
                                "path": sample["path"],
                                "label": chunk,
                                "nb_cols": 1,
                            })

            for i, sample in enumerate(samples):
                ext = sample['path'].split(".")[-1]
                img_name = "{}_{}.{}".format(set_name, i, ext)
                img_path = os.path.join(set_path, img_name)
                #print("sample",sample["label"])
                img = dataset.generate_typed_text_line_image(sample["label"])
                Image.fromarray(img).save(img_path)
                gt[set_name][img_name] = {
                    "text": sample["label"],
                    "nb_cols": sample["nb_cols"] if "nb_cols" in sample else 1
                }
                if "line_label" in sample:
                    gt[set_name][img_name]["lines"] = sample["line_label"]

        with open(os.path.join(path, "labels.pkl"), "wb") as f:
            pickle.dump({
                "ground_truth": gt,
                "charset": sorted(list(charset)),
            }, f)