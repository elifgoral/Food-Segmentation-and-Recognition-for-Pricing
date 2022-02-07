import os
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

OD_RESULT_IMAGES_PATH = os.path.join(CURRENT_PATH, "result_subimages")
COMPARISON_DATASET_PATH = os.path.join(CURRENT_PATH, "comparison_dataset")
PRICE_TABLE_PATH = os.path.join(CURRENT_PATH, "price.csv")

# SIAMESE_MODEL_PATH = os.path.join(CURRENT_PATH, "siamese_models","model5","150_images_150_epoch.h5")
# SIAMESE_MODEL_PATH = os.path.join(CURRENT_PATH, "siamese_models","model6","175_images_150_epoch.h5")
# SIAMESE_MODEL_PATH = os.path.join(CURRENT_PATH, "siamese_models","euclid","1","600_images_150_epoch_binarycross.h5")
SIAMESE_MODEL_PATH = os.path.join(CURRENT_PATH, "siamese_models","euclid","2","600_images_150_epoch_binarycross_2.h5")
# SIAMESE_MODEL_PATH = os.path.join(CURRENT_PATH, "siamese_models","euclid","3","600_images_150_epoch_binarycross_3.h5")


CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

IMAGE_SIZE = 100
THRESHOLD = 0.8

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
}


files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

labels = [{'name':'drink', 'id':1}, {'name':'soup', 'id':2}, {'name':'plate', 'id':3}, {'name':'dessert', 'id':4}, {'name':'salad', 'id':5}]
