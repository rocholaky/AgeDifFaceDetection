from os import listdir
from PIL import Image
from numpy import asarray
import numpy as np
from matplotlib import pyplot
import mtcnn
import tensorflow as tf



def extract_images(path,anchor_list, positive_list, negative_list, size=(160, 160)):

    anchor = from_list_to_images(anchor_list, path, size)
    positive = from_list_to_images(positive_list, path, size)
    negative = from_list_to_images(negative_list, path, size)

    return anchor, positive, negative

def from_list_to_images(alist, path, size):
    the_output_list= list()
    for file in alist:
        image_path = path + "/" + file
        the_output_list.append(get_image(image_path, size))
    return np.vstack(the_output_list)

def get_image(path, size=(160, 160)):
    #get image
    image = Image.open(path)
    #convert to RGB:
    image = image.convert('RGB')
    #resize image:
    image = image.resize(size)
    #turn into array:
    image = asarray(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    return image

def do_normalization(images):
    #compute the mean:
    mean = np.mean(images, axis=0, keepdims=True)
    print(mean.shape)
    std = np.std(images, axis=0, keepdims=True)
    print(std.shape)
    norm_images = (images - mean)
    norm_images = norm_images/std
    return norm_images.astype(np.float32)

    
@tf.function(experimental_relax_shapes=True)
def face_detector(face_dataset):
    output_dataset = list()
    face_dimensions = face_dataset[0].shape[:-1]
    detector = mtcnn.MTCNN()

    for i in range(0, face_dataset.shape[0]):
        image = face_dataset[i][1:]
        results = detector.detect_faces(image)
        if len(results)==0:
           detector = mtcnn.MTCNN()
           results = detector.detect_faces(image)
           
        x1, y1, width, height = results[0]['box']
        x1 = abs(x1)
        y1 = abs(y1)
        x2, y2 = x1 + width, y1+height
        image = image[y1:y2, x1:x2]
        image = Image.fromarray(image)
        image = image.resize((face_dimensions[0], face_dimensions[1]))
        image = np.expand_dims(asarray(image), 0)
        output_dataset.append(image)
    output_dataset = np.vstack(output_dataset)
    return output_dataset
