import os
from tensorflow import keras
import tensorflow as tf
from prettytable import PrettyTable
from datetime import datetime

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
IMAGE_SIZE = 100
THRESHOLD = 0.8

model_paths = {
    "model_1": os.path.abspath(os.path.join(os.path.dirname( __file__ ), "..", "siamese_models","model1","100_images_50_epoch.h5")),
    "model_2": os.path.abspath(os.path.join(os.path.dirname( __file__ ), "..", "siamese_models","model2","100_images_100_epoch.h5")),
    "model_3": os.path.abspath(os.path.join(os.path.dirname( __file__ ), "..", "siamese_models","model3","100_images_150_epoch.h5")),
    "model_4": os.path.abspath(os.path.join(os.path.dirname( __file__ ), "..", "siamese_models","model4","130_images_150_epoch.h5")),
    "model_5": os.path.abspath(os.path.join(os.path.dirname( __file__ ), "..", "siamese_models","model5","150_images_150_epoch.h5")),
    "model_6": os.path.abspath(os.path.join(os.path.dirname( __file__ ), "..", "siamese_models","model6","175_images_150_epoch.h5")),
    "model_e_1": os.path.abspath(os.path.join(os.path.dirname( __file__ ), "..", "siamese_models","euclid","1","600_images_150_epoch_binarycross.h5")),
    "model_e_2": os.path.abspath(os.path.join(os.path.dirname( __file__ ), "..", "siamese_models","euclid","2","600_images_150_epoch_binarycross_2.h5")),
    "model_e_3": os.path.abspath(os.path.join(os.path.dirname( __file__ ), "..", "siamese_models","euclid","3","600_images_150_epoch_binarycross_3.h5")),
}

# That method test the siamese network with two given images and return the similarity of there images(similar or not similar).
def test_siamese(img_path_one, img_path_two, model):
    img_one = tf.io.read_file(img_path_one)
    img_one = tf.image.convert_image_dtype(tf.io.decode_png(img_one, channels=3), dtype='float32')  # * 1./255
    img_one = tf.image.resize(img_one, (IMAGE_SIZE, IMAGE_SIZE), method=tf.image.ResizeMethod.BILINEAR)
    img_one_final = tf.expand_dims(img_one, 0)

    img_two = tf.io.read_file(img_path_two)
    img_two = tf.image.convert_image_dtype(tf.io.decode_png(img_two, channels=3), dtype='float32')  # * 1./255
    img_two = tf.image.resize(img_two, (IMAGE_SIZE, IMAGE_SIZE), method=tf.image.ResizeMethod.BILINEAR)
    img_two_final = tf.expand_dims(img_two, 0)
    
    y_pred= model.predict([img_one_final, img_two_final])
    if y_pred < THRESHOLD:
        return y_pred,"similar"
    else:
        return y_pred,"not similar"


if __name__ == "__main__":


    model_1 = keras.models.load_model(model_paths["model_1"])
    model_2 = keras.models.load_model(model_paths["model_2"])
    model_3 = keras.models.load_model(model_paths["model_3"])
    model_4 = keras.models.load_model(model_paths["model_4"])
    model_5 = keras.models.load_model(model_paths["model_5"])
    model_6 = keras.models.load_model(model_paths["model_6"])
    model_e_1 = keras.models.load_model(model_paths["model_e_1"])
    model_e_2 = keras.models.load_model(model_paths["model_e_2"])
    model_e_3 = keras.models.load_model(model_paths["model_e_3"])

    models = {
        "model_1": model_1,
        "model_2": model_2,
        "model_3": model_3,
        "model_4": model_4,
        "model_5": model_5,
        "model_6": model_6,
        "model_e_1": model_e_1,
        "model_e_2": model_e_2,
        "model_e_3": model_e_3,
    }
    # model.summary()
    img1 = os.path.join(CURRENT_PATH, "Images", "croque_madame_1.jpg")
    img2 = os.path.join(CURRENT_PATH, "Images", "croque_madame_2.jpg")
    img3 = os.path.join(CURRENT_PATH, "Images", "greek_salad_1.jpg")
    img4 = os.path.join(CURRENT_PATH, "Images", "greek_salad_2.jpg")
    img5 = os.path.join(CURRENT_PATH, "Images", "lobster_bisque_1.jpg")
    img6 = os.path.join(CURRENT_PATH, "Images", "lobster_bisque_2.jpg")
    img7 = os.path.join(CURRENT_PATH, "Images", "baby_back_ribs_1.jpg")
    img8 = os.path.join(CURRENT_PATH, "Images", "baby_back_ribs_2.jpg")
    img9 = os.path.join(CURRENT_PATH, "Images", "creme_brulee_1.jpg")
    img10 = os.path.join(CURRENT_PATH, "Images", "creme_brulee_2.jpg")
    
    x = PrettyTable()
    x.field_names = ["Model", "y_pred","Similarity"]   
    start=datetime.now()

    for key in models:
        y_pred, similarity = test_siamese(img1,img3,models[key])
        x.add_row([key, y_pred,similarity])
    print(x)
    print(datetime.now()-start)
