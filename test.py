from model import *
from data import *
import math
#import cv2
#import flwr
#import pytorch as torch

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myG = trainGenerator(2,'data/train','image','label',data_gen_args,save_to_dir = None)

model_checkpoint = ModelCheckpoint('model_test.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myG,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

testG = testGenerator("data/membrane/test")
results = model.predict_generator(testG,30,verbose=1)
