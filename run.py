import os
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2 
import csv
import random
import numpy as np
from tensorflow import keras
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import config
from datetime import datetime
from prettytable import PrettyTable

configs = config_util.get_configs_from_pipeline_file(config.files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(config.paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections    

category_index = label_map_util.create_category_index_from_labelmap(config.files['LABELMAP'])
IMAGE_PATH = os.path.join(config.paths['IMAGE_PATH'], 'test', '42.png')


def find_label_name(id):
    for i in range(len(config.labels)):
        if config.labels[i]["id"] == id:
            return config.labels[i]["name"] 
    return "undefined"

"""
    That method firstly get image from path and convert it to numpy array. Detect the subimages.
    If detected image's detection score is greater than threshold, get the subimages with detection coordinates.
    Then save it under the result_subimages folder with its label. Saved subimage format: label_counter.png
"""
def save_subimages():
    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)


    boxes=detections['detection_boxes'][:len(detections['detection_scores'])]
    image = image_np_with_detections
    width = image.shape[1]
    height = image.shape[0]

    counters = {"soup":0, "drink":0, "plate":0, "dessert":0, "salad":0}
    threshold = 0.6
    for idx , box in enumerate(boxes):
        if detections['detection_scores'][idx] > threshold:
            label_id = detections['detection_classes'][idx]+label_id_offset
            label_name = find_label_name(label_id)
            name = label_name + "_" + str(counters[label_name]) + ".png"
            roi = box*[height,width,height,width]
            region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
            counters[label_name] += 1
            plt.imshow(cv2.cvtColor(region,cv2.COLOR_BGR2RGB))        
            plt.savefig(os.path.join(config.CURRENT_PATH, "result_subimages", name))    

"""
    That method create a price table with randomly generated prices.
"""
def create_price_table():
    counter = 0
    header = ['index', 'class_name', 'price']
    with open(config.PRICE_TABLE_PATH, 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        labels = os.listdir(config.COMPARISON_DATASET_PATH)
        for current_label in labels:
            current_label_path = os.path.join(config.COMPARISON_DATASET_PATH, current_label)
            images = os.listdir(current_label_path)
            for current_image in images:
                class_name = current_image.split(".")[0]
                data = [str(counter),class_name,str(random.randint(10,50))]
                writer.writerow(data)
                counter += 1

"""
    That method get images from result_subimages folder and return image arrays and its labels which is created with object detection.
"""
def get_sub_images():
    images = os.listdir(config.OD_RESULT_IMAGES_PATH)
    image_array = []
    for current_image in images:
        image_path = os.path.join(config.OD_RESULT_IMAGES_PATH, current_image)
        label = current_image.split("_")[0]
        img_dict = {}
        img_dict["image_path"] = image_path
        img_dict["label"] = label
        image_array.append(img_dict)
    return image_array

"""
    Parameters:
    image_path: image path taken from 'get_sub_images' function which will be compared with comparison dataset.
    image_label: image label taken from 'get_sub_images' function which will be compared with comparison dataset.
    
    Algorithm:
    That function compares the images which is taken by first parameter and all the images under the comparision dataset's specified label folder.
    test_siamese method returns the similarity between images. If images are similar, then get_price_from_price_table method will call with label.
    And it will return the price from dataset. (minimmum distance from images)
    Method returns the price and class name 
"""
def get_images_class(image_path,image_label):
    model = keras.models.load_model(config.SIAMESE_MODEL_PATH)
    PATH = os.path.join(config.COMPARISON_DATASET_PATH, image_label)
    images = os.listdir(PATH)
    list_similars = []
    x = PrettyTable()
    x.field_names = ["class", "price","pred"]   
    for current_image in images:
        current_image_path = os.path.join(PATH, current_image)
        similarity, result_pred = test_siamese(current_image_path,image_path,model)
        if similarity == "similar":
            class_name = current_image.split(".")[0]
            price = get_price_from_price_table(class_name)
            food = {"class_name":class_name,"price":price,"result_pred":result_pred}
            list_similars.append(food)
            x.add_row([food['class_name'], food['price'],food['result_pred']])
    print(x)
    min = 10
    for i in range(len(list_similars)):
        if list_similars[i]["result_pred"] < min:
            min = list_similars[i]["result_pred"]
            min_food = list_similars[i]
    if len(list_similars) != 0:
        return min_food["class_name"],min_food["price"]
    return "undefined",0

"""
    That method check the similarity of images with siamese network.
"""
def test_siamese(img_path_one, img_path_two, model):
    img_one = tf.io.read_file(img_path_one)
    img_one = tf.image.convert_image_dtype(tf.io.decode_png(img_one, channels=3), dtype='float32')  # * 1./255
    img_one = tf.image.resize(img_one, (config.IMAGE_SIZE, config.IMAGE_SIZE), method=tf.image.ResizeMethod.BILINEAR)
    img_one_final = tf.expand_dims(img_one, 0)

    img_two = tf.io.read_file(img_path_two)
    img_two = tf.image.convert_image_dtype(tf.io.decode_png(img_two, channels=3), dtype='float32')  # * 1./255
    img_two = tf.image.resize(img_two, (config.IMAGE_SIZE, config.IMAGE_SIZE), method=tf.image.ResizeMethod.BILINEAR)
    img_two_final = tf.expand_dims(img_two, 0)
    
    y_pred= model.predict([img_one_final, img_two_final])
    # print(y_pred)
    if y_pred < config.THRESHOLD:
        return "similar", y_pred
    else:
        return "not_similar", y_pred

"""
    That method gets prices of food classes from price table dataset.
"""
def get_price_from_price_table(class_name):
    with open(config.PRICE_TABLE_PATH, 'r') as file:
        csv_file = csv.reader(file)
        for row in csv_file:
            values = row[0].split(";")
            if values[1] == class_name:
                return values[2]
    return -1

"""
    That method clears the images under the result_subimages folder.
"""
def clear_subimages():
    images = os.listdir(config.OD_RESULT_IMAGES_PATH)
    for current_image in images:
        image_path = os.path.join(config.OD_RESULT_IMAGES_PATH, current_image)
        os.remove(image_path)


if __name__ == "__main__":
    clear_subimages()
    start=datetime.now()
    save_subimages()
    image_array = get_sub_images()
    for i in range(len(image_array)):
        class_name, price = get_images_class(image_array[i]["image_path"],image_array[i]["label"])
        print(f'image path: {image_array[i]["image_path"]}')
        print(f'image label: {image_array[i]["label"]}')
        print(f'image class: {class_name}')
        print(f'image price: {price}')
        print()

    print(datetime.now()-start)
