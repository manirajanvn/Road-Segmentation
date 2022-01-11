#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
from pycocotools.coco import COCO


# In[2]:


import cv2


# In[3]:


annotations = 'D:/ws/dldata/data/road/road_damaged_download/sample/road_damage.json'


# In[4]:


coco_anno=COCO(annotations)


# In[5]:


catIDs = coco_anno.getCatIds()


# In[6]:


catIDs


# In[7]:


imgIds = coco_anno.getImgIds(catIds=catIDs)


# In[8]:


len(imgIds)


# In[9]:


for i in range(len(imgIds)):
  img = coco_anno.loadImgs(imgIds[i])[0]
  file_name = img['file_name'].split('.')[0]
  print(img['file_name'])
  annIds = coco_anno.getAnnIds(imgIds=img['id'], catIds=catIDs, iscrowd=None)
  anns = coco_anno.loadAnns(annIds)
  mask = np.zeros((img['height'],img['width']))
  for i in range(len(anns)):
      mask = np.maximum(coco_anno.annToMask(anns[i]), mask)
  cv2.imwrite('D:/ws/dldata/data/road/road_damaged_download/sample/'+file_name+".png", mask * 255)


# In[ ]:




