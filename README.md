A local-global feature interaction network for fine-grained  
image classification 
Introduction 
Fine-grained visual classification (FGVC) aims at classifying sub-classes of a given object 
category, e.g., subcategories of birds, cars and aircrafts. These are all challenging tasks even to the 
average human and usually require expertise. FGVC is challenging because objects that belong to 
different categories might have similar characteristics, but differences between sub-categories might 
be remarkable (small inter-class variations and large intra-class variations). 
To address these challenges, many deep learning methods have been proposed and can be roughly 
divided into two categories, i.e., part locating methods and feature encoding methods. Part locating 
methods aim to find subtle differences between input images by locating the bounding boxes of 
discriminative parts. These methods generally require expensive manual annotations that limit their 
applications in many scenes. Another category is the feature encoding methods, which aim to learn 
the rich features through a feature fusion strategy. These methods rely too much on the feature 
extraction capabilities of convolutional neural networks. 
Recently, the vision transformer achieved huge success in the classification task which shows that 
applying a pure transformer directly to a sequence of image patches with its innate attention 
mechanism can capture the important regions in images. A series of extended works on downstream 
tasks such as object detection and semantic segmentation confirmed the strong ability for it to 
capture both global and local features. These abilities of the Transformer make it innately suitable 
for the FGVC task as the early long-range “receptive field” of the Transformer enables it to locate 
subtle differences and their spatial relation in the earlier processing layers.  
Although the vision transformer has many advantages in addressing fine-grained classification 
problems, it also faces some limitations and challenges. Such as insufficient handling of local 
information, dependency on large-scale datasets, long training times, high computational and 
memory requirements. In contrast, convolution is better suited for extracting local features, has 
translation invariance, and offers higher parameter and computational efficiency. 




To alleviate the above problems, this study proposes a novel Convolutional Neural Network(CNN) 
and Transformer hybrid network architecture, namely Local-Global Feature Interaction Network 
(LGFINet). The structure of LGFINet is shown in Figure 1. As shown in the figure, the LGFINet 
contains three main parts: the Local Feature Extractor (LFE), the Global Feature Extractor (GFE) 
and the Feature Aggregation Network (FAN). 
Datasets  
(1) CUB-200-2011: Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the 
CUB-200 dataset, with roughly double the number of images per class and new part location 
annotations. (http://www.vision.caltech.edu/datasets/cub_200_2011/)  
Number of categories: 200  
Number of images: 11,788  
Annotations per image: 15 Part Locations, 312 Binary Attributes, 1 Bounding Box  
(2) Stanford Dogs: The Stanford Dogs dataset contains images of 120 breeds of dogs from around 
the world. This dataset has been built using images and annotation from ImageNet for the task of 
fine-grained image categorization. (http://vision.stanford.edu/aditya86/ImageNetDogs/)  
Number of categories: 120  
Number of images: 20,580  
Annotations: Class labels, Bounding boxes  
(3) FGVC-Aircraft: Fine-Grained Visual Classification of Aircraft (FGVC-Aircraft) is a 
benchmark dataset for the fine-grained visual categorization of aircraft. (https://www.robots.ox.ac. 
uk/~vgg/data/fgvc-aircraft/) 
