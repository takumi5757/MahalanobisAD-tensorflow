#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras.layers import Input, GlobalAveragePooling2D
import numpy as np
from keras import Model
import os
from PIL import Image
import cv2

import pandas as pd

from tqdm.notebook import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis

import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as preprocess_input_efficientnet,
)


# In[2]:


class FeatureExtractor(object):
    def __init__(self):
        input_tensor = Input(shape=(224, 224, 3))
        pretrained_model = tf.keras.applications.EfficientNetB0(
            input_tensor=input_tensor, include_top=False, weights="imagenet"
        )

        m1_layer_name = "stem_activation"
        m2_layer_name = "block1a_project_bn"
        m3_layer_name = "block2b_add"
        m4_layer_name = "block3b_add"
        m5_layer_name = "block4c_add"
        m6_layer_name = "block5c_add"
        m7_layer_name = "block6d_add"
        m8_layer_name = "block7a_project_bn"
        m9_layer_name = "top_activation"
        layer_names = [
            m1_layer_name,
            m2_layer_name,
            m3_layer_name,
            m4_layer_name,
            m5_layer_name,
            m6_layer_name,
            m7_layer_name,
            m8_layer_name,
            m9_layer_name,
        ]
        input_tensor = Input(shape=(224, 224, 3))
        pretrained_model = tf.keras.applications.EfficientNetB0(
            input_tensor=input_tensor, include_top=False, weights="imagenet"
        )

        inputs = pretrained_model.inputs
        outputs = []
        for layer_name in layer_names:
            x = pretrained_model.get_layer(layer_name).output
            outputs.append(GlobalAveragePooling2D(data_format="channels_last")(x))

        self.model = Model(inputs=inputs, outputs=outputs)

    def preprocess(self, inputs):
        return preprocess_input_efficientnet(inputs)

    def __call__(self, inputs):
        model = self.model

        feat_list = model(self.preprocess(inputs))

        return feat_list


# In[6]:


class DataLoader(object):
    def __init__(
        self,
        dataset_path="../data/mvtec_anomaly_detection/",
        class_name="bottle",
        is_train=True,
        batch_size=32,
        epochs=1,
    ):

        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.batch_size = batch_size
        self.epochs = epochs

        self.current_index = 0
        self.current_epoch = 1

        self.x, self.y, self.mask = self.load_dataset_folder()
        self.df = pd.DataFrame({"x": self.x, "y": self.y, "mask": self.mask})

    def __iter__(self):
        return self

    def __len__(self):
        return (len(self.x) // self.batch_size) + 1

    def __next__(self):
        if self.epochs is not None and self.current_epoch > self.epochs:
            raise StopIteration()

        df, batch_size, current_index = self.df, self.batch_size, self.current_index
        slice_begin = current_index
        slice_end = current_index + batch_size
        df_batch = df.iloc[slice_begin:slice_end]

        if len(df_batch) != batch_size:
            """
            slice_begin = 0
            slice_end = batch_size - len(df_batch)
            compensational_set = df.iloc[slice_begin:slice_end]
            df_batch = pd.concat([df_batch, compensational_set])
            """
            self.current_epoch += 1

        x_batch, y_batch, mask_batch = self.load_df(df_batch)
        self.current_index = slice_end

        return x_batch, y_batch, mask_batch

    def load_df(self, df_batch):
        x_list = []
        y_list = []
        mask_list = []
        for _, row in df_batch.iterrows():
            x = cv2.imread(row["x"])
            x = cv2.resize(x, (224, 224))

            if row["y"] == 0:
                mask = np.zeros([224, 224, 3])
            else:
                mask = cv2.imread(row["mask"])
                mask = cv2.resize(mask, (224, 224))

            x_list.append(x)
            y_list.append(row["y"])
            mask_list.append(mask)

        return np.asarray(x_list), np.asarray(y_list), np.asarray(mask_list)

    def load_dataset_folder(self):
        phase = "train" if self.is_train else "test"
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, "ground_truth")

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [
                    os.path.join(img_type_dir, f)
                    for f in os.listdir(img_type_dir)
                    if f.endswith(".png")
                ]
            )
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == "good":
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [
                    os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list
                ]
                gt_fpath_list = [
                    os.path.join(gt_type_dir, img_fname + "_mask.png")
                    for img_fname in img_fname_list
                ]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), "number of x and y should be same"

        return list(x), list(y), list(mask)


# In[7]:


model = FeatureExtractor()


# In[8]:


train_dataloader = DataLoader(is_train=True)
test_dataloader = DataLoader(is_train=False)


# In[9]:


class_name = "bottle"
train_outputs = [[] for _ in range(9)]
test_outputs = [[] for _ in range(9)]
for (x, y, mask) in tqdm(
    train_dataloader, "| feature extraction | train | %s |" % class_name
):
    feats = model(x)
    for f_idx, feat in enumerate(feats):
        train_outputs[f_idx].append(feat)


# In[10]:


model_name = "efb0"
save_path = "../results/"
os.makedirs(save_path, exist_ok=True)
train_feat_filepath = os.path.join(
    save_path, "train_%s_%s.pkl" % (class_name, model_name)
)
# fitting a multivariate gaussian to features extracted from every level of ImageNet pre-trained model
for t_idx, train_output in enumerate(train_outputs):
    mean = tf.reduce_mean(tf.concat(train_output, axis=0), axis=0).numpy()
    # covariance estimation by using the Ledoit. Wolf et al. method
    cov = LedoitWolf().fit(tf.concat(train_output, axis=0).numpy()).covariance_
    train_outputs[t_idx] = [mean, cov]

# save extracted feature
with open(train_feat_filepath, "wb") as f:
    pickle.dump(train_outputs, f)


# In[11]:


gt_list = []
# extract test set features
for (x, y, mask) in tqdm(
    test_dataloader, "| feature extraction | test | %s |" % class_name
):
    gt_list.extend(y)
    # model prediction
    feats = model(x)
    for f_idx, feat in enumerate(feats):
        test_outputs[f_idx].append(feat)
for t_idx, test_output in enumerate(test_outputs):
    test_outputs[t_idx] = tf.concat(test_output, axis=0).numpy()


# In[12]:


# calculate Mahalanobis distance per each level of EfficientNet
dist_list = []
predict_list = []
for t_idx, test_output in enumerate(test_outputs):
    mean = train_outputs[t_idx][0]
    cov_inv = np.linalg.inv(train_outputs[t_idx][1])
    dist = [mahalanobis(sample, mean, cov_inv) for sample in test_output]
    dist_list.append(np.array(dist))


# In[13]:


# Anomaly score is followed by unweighted summation of the Mahalanobis distances
scores = np.sum(np.array(dist_list), axis=0)


# In[14]:


# calculate image-level ROC AUC score
total_roc_auc = []
fpr, tpr, thresholds = roc_curve(gt_list, scores)
roc_auc = roc_auc_score(gt_list, scores)
total_roc_auc.append(roc_auc)
print("%s ROCAUC: %.3f" % (class_name, roc_auc))
plt.plot(fpr, tpr, label="%s ROCAUC: %.3f" % (class_name, roc_auc))

print("Average ROCAUC: %.3f" % np.mean(total_roc_auc))
plt.title("Average image ROCAUC: %.3f" % np.mean(total_roc_auc))
plt.legend(loc="lower right")
plt.savefig(os.path.join(save_path, "roc_curve_%s.png" % model_name), dpi=200)


# In[ ]:
