# deeplab_project

Custom training with tensorflow's deeplabv3+ implementation as a Google Colab Jupyter notebook

**Github File descriptions:**
* deeplab.ipynb
    * jupyter notebook to custom train over a face/hair/background segmentation dataset in google colab
* celebA_data.ipynb
    * jupyter notebook to preprocess celebA images and masks 
* modified_files/
    * directory that contains tensorflow's files that need to be modified for custom training with deeplab
* README_images/
    * results images
* Refined_Hair_Image_Segmentation_poster.pdf
    * poster presented at Cal Poly Pomona's Creative Activities & Research Symposium (CARS)
* deeplabv3plus_slides.pdf
    * presentation slides about deeplabv3+ and custom detection

## Custom Training Steps
1. Modify tensorflow files with your dataset information
2. Convert masks to color-indexed images
3. Create
    * ‘train.txt’ with training image filenames
    * ‘val.txt’ with validation image filenames
    * ‘trainval.txt’ with both
    * NOTE: training images and masks should have the same filename
4. Create a tfrecord of your dataset using ‘build_voc2012_data.py’
    * Tfrecord = a Tensorflow binary storage format
5. If using a pre-trained model (transfer learning)
    * https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
6. Run ‘train.py’
7. Run ‘eval.py’ - custom file has miou per class and overall miou
8. Run ‘vis.py’ - to get segmentation masks
9. Run ‘export_model.py’ to save your trained ‘frozen_inference_graph.pb’
10. Use ‘deeplab_demo.ipynb’ with your checkpoint for future inference
   
**These steps can be seen in the deeplab.ipynb jupyter notebook for google colab**    

## Custom Dataset
* (Labeled Faces in the Wild - Parts Dataset)[http://vis-www.cs.umass.edu/lfw/part_labels/]
* (3556 CelebA segmentation masks)[http://www.cs.ubbcluj.ro/~dadi/face-hair-segm-database.html]
  * (CelebA images)[http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html]
  
**Preprocessing needs to be done to convert segmentation maps to color-indexed images.**
* 0 = background
* 1 = hair
* 2 = skin

### Preprocessing LFW - Parts Dataset
The segmentation maps have the following colors converted to 
* blue to 0 = background
* red to 1 = hair
* green to 2 = skin

### Preprocessing CelebA
The file 'celebA_data.ipynb' does the preprocessing seperate from the 'deeplab.ipynb'. We have to convert to color-indexed image and fix some bad pixels in the image.


## Results
Weights for classes:
* 1, background
* 10, hair
* 5, face

**10000 iterations: miou**
* class_0: 0.91722
* class_1: 0.66235
* class_2: 0.83431
* mean_iou: 0.8046266666666666

**20000 iterations: miou**
* class_0: 0.92631
* class_1: 0.67995
* class_2: 0.85498
* mean_iou: 0.8204133333333333

### Good results

![Result Curly Good](/README_images/good/curly3.jpg)
![Result Kinky Good](/README_images/good/kinky3.jpg)
![Result Straight Good](/README_images/good/straight4.jpg)

### Bad results

There are some issues with classifying other parts of the skin as face and not getting a complete hair mask.

![Result Curly Bad](/README_images/bad/curly5.jpg)
![Result Short Bad](/README_images/bad/shortmen9.jpg)




## Files to change for custom dataset training

Look in the modified_files/ directory to see files I download into deeplab.ipynb

* train_utils.py
* data_generator.py
* eval.py


### train_utils.py
*Replace*
```
//original
not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
```
*with*
```
//new
ignore_label_weight = 0
label0_weight = 1           #background
label1_weight = 10          #hair
label2_weight = 5           #skin
    
not_ignore_mask = tf.to_float(tf.equal(scaled_labels, 0)) * label0_weight +  tf.to_float(tf.equal(scaled_labels, 1)) * label1_weight +  tf.to_float(tf.equal(scaled_labels, 2)) * label2_weight +  tf.to_float(tf.equal(scaled_labels, ignore_label)) * ignore_label_weight 
```

### data_generator.py
Add DatasetDescriptor objects 
```
_HAIRSKIN_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 5056,  # num of samples in images/training
        'trainval': 5556,
        'val': 500,  # num of samples in images/validation
    },
    num_classes=4,
    ignore_label=255,
)

_HAIRSKIN_TEST_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'test': 47,
    },
    num_classes=4,
    ignore_label=255,
)
```

Modify the _DATASETS_INFORMATION with the new DatasetDescriptor objects
```
_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'hairskin': _HAIRSKIN_SEG_INFORMATION,
    'hairskin_test': _HAIRSKIN_TEST_INFORMATION,
}
```

### eval.py

If you want miou per class, then add this metric at line 155 located within main()
```
#new metric
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix
cm, update_op_cm = _streaming_confusion_matrix(
    labels, predictions, dataset.num_of_classes, weights=weights)

tf.summary.tensor_summary('confusion_matrix', cm )

#end new metric 
```
so that the code looks like this 

```
# Set ignore_label regions to label 0, because metrics.mean_iou requires
# range of labels = [0, dataset.num_classes). Note the ignore_label regions
# are not evaluated since the corresponding regions contain weights = 0.
labels = tf.where(
    tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

predictions_tag = 'miou'
for eval_scale in FLAGS.eval_scales:
  predictions_tag += '_' + str(eval_scale)
if FLAGS.add_flipped_images:
  predictions_tag += '_flipped'

# Define the evaluation metric.
miou, update_op = tf.metrics.mean_iou(
    predictions, labels, dataset.num_of_classes, weights=weights)
tf.summary.scalar(predictions_tag, miou)

#new metric
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix
cm, update_op_cm = _streaming_confusion_matrix(
    labels, predictions, dataset.num_of_classes, weights=weights)

tf.summary.tensor_summary('confusion_matrix', cm )

#end new metric    

summary_op = tf.summary.merge_all()
summary_hook = tf.contrib.training.SummaryAtEndHook(
    log_dir=FLAGS.eval_logdir, summary_op=summary_op)
hooks = [summary_hook]
```
other custom metrics can be added before tf.summary.merge_all() if desired


## Some resources
* http://hellodfan.com/2018/07/06/DeepLabv3-with-own-dataset/
* https://www.freecodecamp.org/news/how-to-use-deeplab-in-tensorflow-for-object-segmentation-using-deep-learning-a5777290ab6b/

## Authors
* **David Hughes** - *Computational Intelligence Lab in the Department of Computer Science at Cal Poly Pomona*
