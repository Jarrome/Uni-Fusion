import os
import sys
#sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
import clip

import tensorflow.compat.v1 as tf
import tensorflow as tf2

from tqdm import tqdm

import pdb
'''
gpus = tf2.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf2.config.experimental.set_memory_growth(gpu, True)
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[1],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*12)]) # Notice here
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        #tf.config.experimental.set_memory_growth(gpus[1], True)

    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)



clip.available_models()
model, preprocess = clip.load("ViT-L/14@336px")
'''
def f_tx(label_src, device):
#args.label_src = 'plant,grass,cat,stone,other'


    labels = []
    print('** Input label value: {} **'.format(label_src))
    lines = label_src.split(',')
    for line in lines:
        label = line
        labels.append(label)

    outputs = model.net.text_encode(labels, device)
    return outputs
'''

def build_text_embedding(categories):
  run_on_gpu = torch.cuda.is_available()
  with torch.no_grad():
    all_text_embeddings = []
    print("Building text embeddings...")
    for category in tqdm(categories):
      texts = clip.tokenize(category)  #tokenize
      if run_on_gpu:
        texts = texts.cuda(1)
      text_embeddings = model.encode_text(texts)  #embed with text encoder

      text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

      text_embedding = text_embeddings.mean(dim=0)

      text_embedding /= text_embedding.norm()

      all_text_embeddings.append(text_embedding)

    all_text_embeddings = torch.stack(all_text_embeddings, dim=1)

    if run_on_gpu:
      all_text_embeddings = all_text_embeddings.cuda(0)
  return all_text_embeddings.cpu().numpy().T


def f_tx(label_src):
    categories = label_src.split(',')
    feat = build_text_embedding(categories)
    return feat


saved_model_dir = os.path.dirname(__file__)+'/exported_model' #@param {type:"string"}
with tf.device('/GPU:1'):
    openseg_model = tf2.saved_model.load(saved_model_dir, tags=[tf.saved_model.tag_constants.SERVING],)

classes = [
'unannotated',  # 0
'wall',  # 1
'floor',  # 2
'chair',  # 3
'table',  # 4
'desk',  # 5
'bed',  # 6
'bookshelf',  # 7
'sofa',  # 8
'sink',  # 9
'bathtub',  # 10
'toilet',  # 11
'curtain',  # 12
'counter',  # 13
'door',  # 14
'window',  # 15
'shower curtain',  # 16
'refrigerator',  # 17
'picture',  # 18
'cabinet',  # 19
'otherfurniture',  # 20
]



text = classes
text[0] = 'other'
text = ','.join(text) 


text_embedding = f_tx(text)#f_tx('desk,table')
num_words_per_category = 1
with tf.device('/GPU:1'):
    text_embedding = tf.reshape(
                  text_embedding, [-1, num_words_per_category, text_embedding.shape[-1]])
    text_embedding = tf.cast(text_embedding, tf.float32)

def f_im(np_str,H=320,W=240):
    with tf.device('/GPU:1'):
        output = openseg_model.signatures['serving_default'](
            inp_image_bytes=tf.convert_to_tensor(np_str[0]),
            inp_text_emb=text_embedding)
        #feat =  output['image_embedding_feat'][0,:480,:,:] # 1,640,640,768 -> 480,640,768

        # if scannet
        '''
        feat =  output['ppixel_ave_feat'][0,:480,:,:] # 1,640,640,768 -> 480,640,768
        feat = tf.image.resize(feat, [240, 320]) # 240,320,768
        '''
        # if 2D-3D-S
        feat = output['ppixel_ave_feat'][0,:,:,:]
        feat_h, feat_w = feat.shape[:2]
        H_ov_W = float(H)/float(W)
        feat_cropped = feat[:int(H_ov_W*feat_h),:]

        feat = tf.image.resize(feat_cropped, [H,W])


        #feat = tf.image.resize(feat, [320, 320])

        feat = feat.numpy()
        #feat = feat / np.linalg.norm(feat, axis=-1, keepdims=True)

    return feat

def classify(image_features, text_features):
    '''
        both in np
        F_im is N,c
        F_tx is k,c where k is the classes
    '''
    #image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logits_per_image = image_features.half() @ text_features.T # N,k
    return logits_per_image

def get_api():
    return f_im, f_tx, classify, 768





'''
For each category you can list different names. Use ';' to separate different categories and use ',' to separate different names of a category.
E.g. 'lady, ladies, girl, girls; book' creates two categories of 'lady or ladies or girl or girls' and 'book'.
'''

