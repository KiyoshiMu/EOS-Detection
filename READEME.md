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

## 2. Unet

###TODO

## 3. Usage

1. point_label_creator

2. point_to_maskor

3. tile_creator

4. Unet_trainor

5. use_model