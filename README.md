Project for EOS detection
===

## Construction

### 1. Kindle

For preparation, we labeled 7 H&E slices. We marked the cells in these images by green points, which are easy to be implemented and dectected by both computers and human beings. 

**Cell_segment.py** harnessed openCV's *cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU* and *watershed* to detect cells and then based on the point labels to segment them into 96 * 96 small individul cells.

**Cell_traning.py** constructed a simple CNN model on Keras powered by Tensorflow. The CNN learned the individul cell images we created in the previous step. Then, it could roughly classify the cell type (whether it's a EOS) on a cell image.

We wanted to acclerate the advance of this project. A bottleneck is the limited labor we had for label. More importantly, labeling images is quite boring. To remove the bottleneck and avoid boredom, we used the simple CNN model to help us. That is the content of **Cell_recognition.py**. After the openCV segmented possible cell areas, the CNN would label the areas by its judgement. It marked EOS it classified by black box. We corrected its work, then. If a label was right, we left it there. Otherwise, we marked a green point inside the black box. Besides, we also mark some cells it didn't notice, by green point. As a result, the label areas are which only in black boxes or only have green points. The suffix of labeled images is **_label**.

At that moment, we had a huge mistake in design, that is we let the black boxes as markers, which can easily overlap with the near and make the labeled images hard to be used by other models or reused. It's because this kind of markers don't clearly and simply tell what is the interest. Therefore, we added another type of labeled images for human-label and machine-label integration. This type uses point markers with 5 pixels as radius and green as color. The suffix of this type is **_help**.

**Cell_refine.py** read the special labels the CNN and we created before. Then, the CNN learned these new data. As a result, it could work better in helping us label. We could generate more data in a certain period of time, in this way, and it can learn more. As a result, it learned more, and we can generate more data. A virtuous circle emerged.

However, this method would come to its limitation somehow. A cutting-edge technology is needed.

### 2. Unet

The [__**paper**__](https://arxiv.org/abs/1505.04597) was Submitted on 18 May 2015 by authors -- Olaf Ronneberger, Philipp Fischer, Thomas Brox. Following is the abstract 
--*There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU.*

![alt text](readme_imgs/u-net-architecture.png "u-net-architecture")

The origin version of Unet powered by Keras may be [this](https://github.com/zhixuhao/unet), which was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/). 

Our version looks like below.

![created by tensorboard](readme_imgs/graph_run.png)

Also, I have pretrain the model used the data from [Kaggle](https://www.kaggle.com/c/data-science-bowl-2018): _2018 Data Science Bowl -- Find the nuclei in divergent images to advance medical discovery._ However, The result has showed this dataset cannot help to improve our model's performance. The lowest *val_loss* of __from stracth__ model is 0.02164, while that of __on kaggle dataset__ model is 0.02295. The differnece is not evident. Still, Many functions are adaptations from its notebooks, including [nuclei-overview-to-submission](https://www.kaggle.com/kmader/nuclei-overview-to-submission/notebook), [keras-u-net-starter](https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277/notebook), [identification-and-segmentation-of-nuclei-in-cells](https://www.kaggle.com/paultimothymooney/identification-and-segmentation-of-nuclei-in-cells) and [basic-pure-computer-vision-segmentation](https://www.kaggle.com/gaborvecsei/basic-pure-computer-vision-segmentation-lb-0-229). Their insights did a great help. **Thank!**

### 3. Training

__Now, the following part 1, 2, 3 is combined into a pipeline *preparation.py*.__

1. point_label_creator
---
Input: a raw images directory path; a labeled images directory path; a directory path for saving
Return: point label **.png** files in saving directory
---
The aim of this part is to compromise with the deficits of a previous bad design that we use the boxed mared images to assist the label progress. However, when it comes to reuse these labeled images, boxes and masks don't cooperate well. It's too large to identify a single area. In contrary, points can do a good job in this work. As a result, we decide to convert the black boxes to green points, in order to make masks by labeled images.

Now, we will generate two version of assistant prediction, a circle labeled version and a point labeled version. The circle labeled version can help the researchers detect and correct the mistakes the model made. It's just because a circle is easy to be seen by nake eyes. The point labeled version can record the prediction the model made and it will not confuse the openCV, when the openCV creats masks by labeled images. _As a result, this part may be no longer useful._

2. point_to_maskor
---
Input: a raw images directory path; a point labeled images directory path; a directory path for saving
Return: mask **.png** files in saving directory
---
The aim of this part is to convert green points to masks. Unet uses masks as its outcomes. To train a Unet model, enough masks are necessary.

The downside of this part is that we use watershed algorithm to segment areas in the first place. Then, the quality of the masks is unstable. However, a large number of training images can circumvent this weakness. Moreover, we want to make the Unet model can learn as many scenes as possibel.

3. tile_creator
---
Input: a raw images directory path; a mask images directory path; a directory path for saving
Return: tile **.png** files in saving directory
---

The information in H&E slides is important, so we decide not to limit it at the beginning, that is not to resize the images. The aim of this part is to separate large images into small images. The original images are usually 1920 * 1440. We create 256 * 256 small images from them. We make the images contain overlaps to contain virgin information from the original images. The number of the small images is to make the overall overlap areas as small as possible.

4. Unet_trainor
---
Input: tiles for traing directory path; tiles for test directory path; model name; retrained model path
__(The model with *.h5* suffix is saved in the director *unet_tools* for *use_model.py*'s future use.)__
_(If the "tiles for test directory path" is "No", this stript will ctrate train dataset and test datasat from the tiles in the "tiles for traing directory".)_
_(If "the retrained model path" is not provided, the stript will generate model from stratch.)_
Return: tile **.png** files in saving directory
---
This part can train our model from scratch or retrain the previous existing model. The function is quite simple. See more information in **2.Unet part**.

## Usage
