Project for EOS detection
===

## Construction

### 1. Kindle

For preparation, we labeled 7 H&E slices. We marked the cells in these images by green points, which are easy to be implemented and dectected by both computers and human beings. 

__Cell_segment.py__ harnessed openCV's *cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU* and *watershed* to detect cells and then based on the point labels to segment them into 96 * 96 small individul cells.

__Cell_traning.py__ constructed a simple CNN model on Keras powered by Tensorflow. The CNN learned the individul cell images we created in the previous step. Then, it could roughly classify the cell type (whether it's a EOS) on a cell image.

We wanted to acclerate the advance of this project. A bottleneck is the limited labor we had for label. More importantly, labeling images is quite boring. To remove the bottleneck and avoid boredom, we used the simple CNN model to help us. That is the content of __Cell_recognition.py__. After the openCV segmented possible cell areas, the CNN would label the areas by its judgement. It marked EOS it classified by black box. We corrected its work, then. If a label was right, we left it there. Otherwise, we marked a green point inside the black box. Besides, we also mark some cells it didn't notice, by green point. As a result, the label areas are which only in black boxes or only have green points.

__Cell_refine.py__ read the special labels the CNN and we created before. Then, the CNN learned these new data. As a result, it could work better in helping us label. We could generate more data in a certain period of time, in this way, and it can learn more. As a result, it learned more, and we can generate more data. A virtuous circle emerged.

However, this method would come to its limitation somehow. A cutting-edge technology is needed.

### 2. Unet

The [__**paper**__](https://arxiv.org/abs/1505.04597) was Submitted on 18 May 2015 by authors -- Olaf Ronneberger, Philipp Fischer, Thomas Brox. Following is the abstract 
--*There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU.*

![readme_imgs/u-net-architecture.png](readme_imgs/u-net-architecture.png)

The origin version of Unet powered by Keras may be [this](https://github.com/zhixuhao/unet), which was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/). 

Our version looks like below.

Also, I have pretrain the model used the data from [Kaggle](https://www.kaggle.com/c/data-science-bowl-2018): _2018 Data Science Bowl -- Find the nuclei in divergent images to advance medical discovery._ The result has showed this dataset can slightly improve our model's performance, about 0.1% in accuracy. Beside, Many functions are adaptations from its notebooks, including [nuclei-overview-to-submission](https://www.kaggle.com/kmader/nuclei-overview-to-submission/notebook), [keras-u-net-starter](https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277/notebook), [identification-and-segmentation-of-nuclei-in-cells](https://www.kaggle.com/paultimothymooney/identification-and-segmentation-of-nuclei-in-cells) and [basic-pure-computer-vision-segmentation](https://www.kaggle.com/gaborvecsei/basic-pure-computer-vision-segmentation-lb-0-229). Their insights did a great help. **Thank!**

### 3. Training

1. point_label_creator

The aim of this part is to compromise with the deficits of a previous bad design that we use the boxed mared images to assist the label progress. However, when it comes to reuse these labeled images, boxes and masks don't cooperate well. It's too large to identify a single area. In contrary, points can do a good job in this work. As a result, we decide to convert the black boxes to green points, in order to make masks by labeled images.

Now, we will generate two version of assistant prediction, a circle labeled version and a point labeled version. The circle labeled version can help the researchers detect and correct the mistakes the model made. It's just because a circle is easy to be seen by nake eyes. The point labeled version can record the prediction the model made and it will not confuse the openCV, when the openCV creats masks by labeled images.

2. point_to_maskor

The aim of this part is to convert green points to masks. Unet uses masks as its outcomes. To train a Unet model, enough masks are necessary.

The downside of this part is that we use watershed algorithm to segment areas in the first place. Then, the quality of the masks is unstable. However, a large number of training images can circumvent this weakness. Moreover, we want to make the Unet model can learn as many scenes as possibel.

3. tile_creator

The information in H&E slides is important, so we decide not to limit it at the beginning, that is not to resize the images. The aim of this part is to separate large images into small images. The original images are usually 1920 * 1440. We create 256 * 256 small images from them. We make the images contain overlaps to contain virgin information from the original images. The number of the small images is to make the overall overlap areas as small as possible.

4. Unet_trainor

This part can train our model from scratch or retrain the previous existing model. The function is quite simple. See more information in __2.Unet part.__.

## Usage
