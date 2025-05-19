from basic.generic_dataset_manager import DatasetManager, GenericDataset
from basic.utils import pad_images, pad_image_width_right, resize_max, pad_image_width_random, pad_sequences_1D, pad_image_height_random, pad_image_width_left, pad_image
from basic.utils import randint, rand, rand_uniform
from basic.generic_dataset_manager import apply_preprocessing
from OCR.ocr_utils import LM_str_to_ind
import random
import cv2
import os
import copy
import pickle
import numpy as np
import torch
import matplotlib
from PIL import Image, ImageDraw, ImageFont
from basic.transforms import RandomRotation, apply_transform, Tightening
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode


class OCRDatasetManager(DatasetManager):
    """
    Specific class to handle OCR/HTR tasks
    excluding any control or special characters that are not part of the actual text content. This ensures the synthetic data is as close as possible to real-world data the OCR model 

    CTC is a type of output layer used in neural networks, particularly in sequence prediction problems without a pre-defined alignment between the input and the output sequences. It's widely used in tasks like speech and handwriting recognition, where the length of the output sequence is not fixed and does not directly align with the input sequence length.
    Seq2Seq is a model architecture designed to map sequences of variable lengths to other sequences of variable lengths. It's particularly useful for translation, chatbots, and any task requiring complex transformations from one sequence to another.
    """

    def __init__(self, params):
        super(OCRDatasetManager, self).__init__(params)

        self.charset = params["charset"] if "charset" in params else self.get_merged_charsets()
        
        if "new_tokens" in params:
            self.charset = sorted(list(set(self.charset).union(set(params["new_tokens"]))))

        self.tokens = {
            "pad": params["config"]["padding_token"],
        }
        if self.params["config"]["charset_mode"].lower() == "ctc":
            self.tokens["blank"] = len(self.charset)
            self.tokens["pad"] = self.tokens["pad"] if self.tokens["pad"] else len(self.charset) + 1
            self.params["config"]["padding_token"] = self.tokens["pad"]
        elif self.params["config"]["charset_mode"] == "seq2seq":
            self.tokens["end"] = len(self.charset)
            self.tokens["start"] = len(self.charset) + 1
            self.tokens["pad"] = self.tokens["pad"] if self.tokens["pad"] else len(self.charset) + 2
            self.params["config"]["padding_token"] = self.tokens["pad"]

    def get_merged_charsets(self):
        """
        Merge the charset of the different datasets used
        """
        datasets = self.params["datasets"]
        charset = set()
        for key in datasets.keys():
            with open(os.path.join(datasets[key], "labels.pkl"), "rb") as f:
                info = pickle.load(f)
                dataset_charset = set(info.get("charset", []))
                charset = charset.union(dataset_charset)
        if "\n" in charset and "remove_linebreaks" in self.params["config"]["constraints"]:
            charset.remove("\n")
        if "" in charset:
            charset.remove("")
        return sorted(list(charset))




class OCRDataset(GenericDataset):
    """
    Specific class to handle OCR/HTR datasets
    """

    def __init__(self, params, set_name, custom_name, paths_and_sets):
        super(OCRDataset, self).__init__(params, set_name, custom_name, paths_and_sets)
        self.charset = None
        self.tokens = None
        self.reduce_dims_factor = np.array([params["config"]["height_divisor"], params["config"]["width_divisor"], 1])
        self.collate_function = OCRCollateFunction
        self.synthetic_id = 0
        
    def __getitem__(self, idx):
        sample = copy.deepcopy(self.samples[idx])

        if not self.load_in_memory:
            sample["img"] = self.get_sample_img(idx)
            sample = apply_preprocessing(sample, self.params["config"]["preprocessings"])

        # Data augmentation
        sample["img"], sample["applied_da"] = self.apply_data_augmentation(sample["img"])

        if "max_size" in self.params["config"] and self.params["config"]["max_size"]:
            max_ratio = max(sample["img"].shape[0] / self.params["config"]["max_size"]["max_height"], sample["img"].shape[1] / self.params["config"]["max_size"]["max_width"])
            if max_ratio > 1:
                new_h, new_w = int(np.ceil(sample["img"].shape[0] / max_ratio)), int(np.ceil(sample["img"].shape[1] / max_ratio))
                sample["img"] = cv2.resize(sample["img"], (new_w, new_h))

        # Normalization if requested
        if "normalize" in self.params["config"] and self.params["config"]["normalize"]:
            sample["img"] = (sample["img"] - self.mean) / self.std

        sample["img_shape"] = sample["img"].shape
        sample["img_reduced_shape"] = np.ceil(sample["img_shape"] / self.reduce_dims_factor).astype(int)

        # Padding to handle CTC requirements
        if self.set_name == "train":
            max_label_len = 0
            height = 1
            ctc_padding = False
            if "CTC_line" in self.params["config"]["constraints"]:
                max_label_len = sample["label_len"]
                ctc_padding = True
            if "CTC_va" in self.params["config"]["constraints"]:
                max_label_len = max(sample["line_label_len"])
                ctc_padding = True
            if "CTC_pg" in self.params["config"]["constraints"]:
                max_label_len = sample["label_len"]
                height = max(sample["img_reduced_shape"][0], 1)
                ctc_padding = True
            if ctc_padding and 2 * max_label_len + 1 > sample["img_reduced_shape"][1]*height:
                sample["img"] = pad_image_width_right(sample["img"], int(np.ceil((2 * max_label_len + 1) / height) * self.reduce_dims_factor[1]), self.padding_value)
                sample["img_shape"] = sample["img"].shape
                sample["img_reduced_shape"] = np.ceil(sample["img_shape"] / self.reduce_dims_factor).astype(int)
            sample["img_reduced_shape"] = [max(1, t) for t in sample["img_reduced_shape"]]

        sample["img_position"] = [[0, sample["img_shape"][0]], [0, sample["img_shape"][1]]]
        # Padding constraints to handle model needs
        if "padding" in self.params["config"] and self.params["config"]["padding"]:
            if self.set_name == "train" or not self.params["config"]["padding"]["train_only"]:
                min_pad = self.params["config"]["padding"]["min_pad"]
                max_pad = self.params["config"]["padding"]["max_pad"]
                pad_width = randint(min_pad, max_pad) if min_pad is not None and max_pad is not None else None
                pad_height = randint(min_pad, max_pad) if min_pad is not None and max_pad is not None else None

                sample["img"], sample["img_position"] = pad_image(sample["img"], padding_value=self.padding_value,
                                          new_width=self.params["config"]["padding"]["min_width"],
                                          new_height=self.params["config"]["padding"]["min_height"],
                                          pad_width=pad_width,
                                          pad_height=pad_height,
                                          padding_mode=self.params["config"]["padding"]["mode"],
                                          return_position=True)
        sample["img_reduced_position"] = [np.ceil(p / factor).astype(int) for p, factor in zip(sample["img_position"], self.reduce_dims_factor[:2])]
        return sample


    def get_charset(self):
        charset = set()
        for i in range(len(self.samples)):
            charset = charset.union(set(self.samples[i]["label"]))
        return charset

    def convert_labels(self):
        """
        Label str to token at character level
        """
        for i in range(len(self.samples)):
            self.samples[i] = self.convert_sample_labels(self.samples[i])

    def convert_sample_labels(self, sample):
        label = sample["label"]
        line_labels = label.split("\n")
        if "remove_linebreaks" in self.params["config"]["constraints"]:
            full_label = label.replace("\n", " ").replace("  ", " ")
            word_labels = full_label.split(" ")
        else:
            full_label = label
            word_labels = label.replace("\n", " ").replace("  ", " ").split(" ")

        sample["label"] = full_label
        sample["token_label"] = LM_str_to_ind(self.charset, full_label)
        if "add_eot" in self.params["config"]["constraints"]:
            sample["token_label"].append(self.tokens["end"])
        sample["label_len"] = len(sample["token_label"])
        if "add_sot" in self.params["config"]["constraints"]:
            sample["token_label"].insert(0, self.tokens["start"])

        sample["line_label"] = line_labels
        sample["token_line_label"] = [LM_str_to_ind(self.charset, l) for l in line_labels]
        sample["line_label_len"] = [len(l) for l in line_labels]
        sample["nb_lines"] = len(line_labels)

        sample["word_label"] = word_labels
        sample["token_word_label"] = [LM_str_to_ind(self.charset, l) for l in word_labels]
        sample["word_label_len"] = [len(l) for l in word_labels]
        sample["nb_words"] = len(word_labels)
        return sample
    
    def generate_typed_text_line_image(self, text):
        #print("self.params",self.params["config"]["synthetic_data"]["config"])
        return generate_typed_text_line_image(text, self.params["config"]["synthetic_data"]["config"])

    def generate_typed_text_paragraph_image(self, texts, padding_value=255, max_pad_left_ratio=0.1, same_font_size=False):
        config = self.params["config"]["synthetic_data"]["config"]
        if same_font_size:
            images = list()
            txt_color = config["text_color_default"]
            bg_color = config["background_color_default"]
            font_size = randint(config["font_size_min"], config["font_size_max"] + 1)
            for text in texts:
                font_path = config["valid_fonts"][randint(0, len(config["valid_fonts"]))]
                fnt = ImageFont.truetype(font_path, font_size)
                text_width, text_height = fnt.getsize(text)
                padding_top = int(rand_uniform(config["padding_top_ratio_min"], config["padding_top_ratio_max"]) * text_height)
                padding_bottom = int(rand_uniform(config["padding_bottom_ratio_min"], config["padding_bottom_ratio_max"]) * text_height)
                padding_left = int(rand_uniform(config["padding_left_ratio_min"], config["padding_left_ratio_max"]) * text_width)
                padding_right = int(rand_uniform(config["padding_right_ratio_min"], config["padding_right_ratio_max"]) * text_width)
                padding = [padding_top, padding_bottom, padding_left, padding_right]
                images.append(generate_typed_text_line_image_from_params(text, fnt, bg_color, txt_color, config["color_mode"], padding))
        else:
            images = [self.generate_typed_text_line_image(t) for t in texts]

        max_width = max([img.shape[1] for img in images])

        padded_images = [pad_image_width_random(img, max_width, padding_value=padding_value, max_pad_left_ratio=max_pad_left_ratio) for img in images]
        return np.concatenate(padded_images, axis=0)



class OCRCollateFunction:
    """
    Merge samples data to mini-batch data for OCR task
    """

    def __init__(self, config):
        self.img_padding_value = float(config["padding_value"])
        self.label_padding_value = config["padding_token"]
        self.config = config

    def __call__(self, batch_data):
        names = [batch_data[i]["name"] for i in range(len(batch_data))]
        ids = [int(batch_data[i]["name"].split("/")[-1].split("_")[-1].split(".")[0]) for i in range(len(batch_data))] 

        applied_da = [batch_data[i]["applied_da"] for i in range(len(batch_data))]

        labels = [batch_data[i]["token_label"] for i in range(len(batch_data))]
        labels = pad_sequences_1D(labels, padding_value=self.label_padding_value)
        labels = torch.tensor(labels).long()
        reverse_labels = [[batch_data[i]["token_label"][0], ] + batch_data[i]["token_label"][-2:0:-1] + [batch_data[i]["token_label"][-1], ] for i in range(len(batch_data))]
        reverse_labels = pad_sequences_1D(reverse_labels, padding_value=self.label_padding_value)
        reverse_labels = torch.tensor(reverse_labels).long()
        labels_len = [batch_data[i]["label_len"] for i in range(len(batch_data))]

        raw_labels = [batch_data[i]["label"] for i in range(len(batch_data))]
        unchanged_labels = [batch_data[i]["unchanged_label"] for i in range(len(batch_data))]

        nb_cols = [batch_data[i]["nb_cols"] for i in range(len(batch_data))]
        nb_lines = [batch_data[i]["nb_lines"] for i in range(len(batch_data))]
        line_raw = [batch_data[i]["line_label"] for i in range(len(batch_data))]
        line_token = [batch_data[i]["token_line_label"] for i in range(len(batch_data))]
        pad_line_token = list()
        line_len = [batch_data[i]["line_label_len"] for i in range(len(batch_data))]
        for i in range(max(nb_lines)):
            current_lines = [line_token[j][i] if i < nb_lines[j] else [self.label_padding_value] for j in range(len(batch_data))]
            pad_line_token.append(torch.tensor(pad_sequences_1D(current_lines, padding_value=self.label_padding_value)).long())
            for j in range(len(batch_data)):
                if i >= nb_lines[j]:
                    line_len[j].append(0)
        line_len = [i for i in zip(*line_len)]

        nb_words = [batch_data[i]["nb_words"] for i in range(len(batch_data))]
        word_raw = [batch_data[i]["word_label"] for i in range(len(batch_data))]
        word_token = [batch_data[i]["token_word_label"] for i in range(len(batch_data))]
        pad_word_token = list()
        word_len = [batch_data[i]["word_label_len"] for i in range(len(batch_data))]
        for i in range(max(nb_words)):
            current_words = [word_token[j][i] if i < nb_words[j] else [self.label_padding_value] for j in range(len(batch_data))]
            pad_word_token.append(torch.tensor(pad_sequences_1D(current_words, padding_value=self.label_padding_value)).long())
            for j in range(len(batch_data)):
                if i >= nb_words[j]:
                    word_len[j].append(0)
        word_len = [i for i in zip(*word_len)]

        padding_mode = self.config["padding_mode"] if "padding_mode" in self.config else "br"
        imgs = [batch_data[i]["img"] for i in range(len(batch_data))]
        imgs_shape = [batch_data[i]["img_shape"] for i in range(len(batch_data))]
        imgs_reduced_shape = [batch_data[i]["img_reduced_shape"] for i in range(len(batch_data))]
        imgs_position = [batch_data[i]["img_position"] for i in range(len(batch_data))]
        imgs_reduced_position= [batch_data[i]["img_reduced_position"] for i in range(len(batch_data))]
        imgs = pad_images(imgs, padding_value=self.img_padding_value, padding_mode=padding_mode)
        imgs = torch.tensor(imgs).float().permute(0, 3, 1, 2)
        formatted_batch_data = {
            "names": names,
            "ids": ids,
            "nb_lines": nb_lines,
            "nb_cols": nb_cols,
            "labels": labels,
            "reverse_labels": reverse_labels,
            "raw_labels": raw_labels,
            "unchanged_labels": unchanged_labels,
            "labels_len": labels_len,
            "imgs": imgs,
            "imgs_shape": imgs_shape,
            "imgs_reduced_shape": imgs_reduced_shape,
            "imgs_position": imgs_position,
            "imgs_reduced_position": imgs_reduced_position,
            "line_raw": line_raw,
            "line_labels": pad_line_token,
            "line_labels_len": line_len,
            "nb_words": nb_words,
            "word_raw": word_raw,
            "word_labels": pad_word_token,
            "word_labels_len": word_len,
            "applied_da": applied_da
        }

        return formatted_batch_data


def generate_typed_text_line_image(text, config, bg_color=(255, 255, 255), txt_color=(0, 0, 0)):
    if text == "":
        text = " "
    if "text_color_default" in config:
        txt_color = config["text_color_default"]
    if "background_color_default" in config:
        bg_color = config["background_color_default"]

    #print("config",config["valid_fonts"])
    font_path = config["valid_fonts"][randint(0, len(config["valid_fonts"]))]
    font_size = randint(config["font_size_min"], config["font_size_max"]+1)
    fnt = ImageFont.truetype(font_path, font_size)

    #text_width, text_height = fnt.getsize(text)
    text_bbox = fnt.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]  # Width is the difference between right and left
    text_height = text_bbox[3] - text_bbox[1]
    padding_top = int(rand_uniform(config["padding_top_ratio_min"], config["padding_top_ratio_max"])*text_height)
    padding_bottom = int(rand_uniform(config["padding_bottom_ratio_min"], config["padding_bottom_ratio_max"])*text_height)
    padding_left = int(rand_uniform(config["padding_left_ratio_min"], config["padding_left_ratio_max"])*text_width)
    padding_right = int(rand_uniform(config["padding_right_ratio_min"], config["padding_right_ratio_max"])*text_width)
    padding = [padding_top, padding_bottom, padding_left, padding_right]
    return generate_typed_text_line_image_from_params(text, fnt, bg_color, txt_color, config["color_mode"], padding)


def generate_typed_text_line_image_from_params(text, font, bg_color, txt_color, color_mode, padding):
    padding_top, padding_bottom, padding_left, padding_right = padding
    #text_width, text_height = font.getsize(text)
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]  # Width is the difference between right and left
    text_height = text_bbox[3] - text_bbox[1]
    img_height = padding_top + padding_bottom + text_height
    img_width = padding_left + padding_right + text_width
    img = Image.new(color_mode, (img_width, img_height), color=bg_color)
    d = ImageDraw.Draw(img)
    d.text((padding_left, padding_bottom), text, font=font, fill=txt_color, spacing=0)
    return np.array(img)


def get_valid_fonts(alphabet=None):
    valid_fonts = list()
    #print("valid_fonts",valid_fonts)
    for fold_detail in os.walk("Fonts"):
        #print("fold_detail",fold_detail[2])
        if fold_detail[2]:
            for font_name in fold_detail[2]:
                if ".ttf" not in font_name:
                    continue
                font_path = os.path.join(fold_detail[0], font_name)
                to_add = True
                if alphabet is not None:
                    for char in alphabet:
                        if not char_in_font(char, font_path):
                            to_add = False
                            break
                    if to_add:
                        valid_fonts.append(font_path)
                else:
                    valid_fonts.append(font_path)
    return valid_fonts


def char_in_font(unicode_char, font_path):
    with TTFont(font_path) as font:
        for cmap in font['cmap'].tables:
            if cmap.isUnicode():
                if ord(unicode_char) in cmap.cmap:
                    return True
    return False