from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from yad2k.utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes
from yad2k.yolo_v2 import yolo_head, yolo_eval
import tensorflow as tf
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--classes', type=str, help='Class names text file location.',
                    default="model_data/coco_classes.txt")
parser.add_argument('-a', '--anchors', type=str, help='Anchors text file location.',
                    default="model_data/yolo_anchors.txt")
parser.add_argument('-m', '--model', type=str, help='Load a pretrained model from h5 file',
                    default="model_data/yolo.h5")
parser.add_argument('-i', '--image_file', type=str, help='Test image file',
                    default="images/test.jpg")

args = parser.parse_args()

if __name__ == '__main__':
    sess = K.get_session()

    class_names = read_classes(args.classes)
    anchors = read_anchors(args.anchors)
    image_shape = tf.constant([720., 1280.])

    yolo_model = load_model(args.model)

    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
    image, image_data = preprocess_image("images/" + args.image_file, model_image_size=(608, 608))
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    colors = generate_colors(len(class_names))
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.save(os.path.join("out", args.image_file), quality=90)