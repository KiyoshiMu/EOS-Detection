Project for EOS detection
=======

## Construction

**HISTORY**

### 1. Kindle

For preparation, we labeled 7 H&E slices. We marked the cells in these images by green points, which are easy to be implemented and dectected by both computers and human beings. 

Cell_segment.py harnessed openCV's cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU and watershed to detect cells and then based on the point labels segmented them into 96 * 96 small individul cells.

Cell_traning.py constructed a simple CNN model on Keras that is on tensorflow. The CNN learned the cell images we created in the previous step. Then, it could roughly classify the cell type (whether it's a EOS) on a cell image.

We wanted to acclerate the advance of this project. Part of the bottleneck is the labor for label. In order to reduce the label work, We used the simple CNN model to help us. That is the content of Cell_recognition.py. After the openCV segmented possible cell areas, the CNN would label the areas by its judgement. It marked the EOS it thought on slices by black box. We corrected its work, then. If a label was right, we left it there. Otherwise, we marked a green point inside the black box. Besides, we also mark some cells it didn't notice by green point. As a result, the label areas are which only in black boxes or only have green points.

Cell_refine.py read the special labels the CNN and we created before. Then, the CNN learned these new data. As a result, it could work better in helping us label. We could generate more data in a certain period of time, in this way, and it can learn more. Also, it learned more, and we can generate data faster. A virtuous circle emerged.

However, this method will come to its limitation. A cutting-edge technology is needed.

### 2. Unet

TODO

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



## Usage
