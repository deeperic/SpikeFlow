# SpikeFlow
A Chinese OCR with TensorFlow

*** Warning: The source codes in this repository may not work well with the latest version of Tensorflow. ***

To play around, follow these steps:

1/ Use Ocropy to generate Chinese character images. linegen is the tool used. You will need a font file. Put the images under folder training-character.gh.

2/ Run labelling-character.py to generate the labels on images.

3/ Run tf/convert-to-tfrecords.py to convert the images and labels in Tensorflow format.

4/ Modify the tf/helper.py for the characters you want to recognise.

5/ Run tf/train_model.py to train a model. The training will save a checkpoint on a regular interval.

6/ In receipt, run the find_contour_character.py to generate the images which may contain Chinese characters. You will have a "bw" folder containing all images. Run: python find_contour_character.py {image filename}

7/ Test the model by running: python test_one_char.py {the name of your model} {the image to be recognised}


Blog: https://deeperic.wordpress.com/2017/02/18/chinese-ocr-tensorflow

Youtube: https://youtu.be/9N9OUruPZd4

GitHub: https://github.com/deeperic/SpikeFlow
