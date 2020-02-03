import os
import numpy as np
from voc_annotation import VOCAnnotation
from voc_data import VocData
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model.model import yolo_model
import config as cfg
from loss_func import yolo_loss


voc_annotation = VOCAnnotation('~/segment_data', 2007, 'train', './data/voc_classes.txt')
annotations = voc_annotation.get_annotations()

samples_num = len(annotations)
val_split = 0.1
val_num = int(samples_num * val_split)
train_num = samples_num - val_num

voc_data = VocData()
voc_data_g = voc_data.data_generator_wrapper(annotations[:train_num], batch_size=20)
val_data_g = voc_data.data_generator_wrapper(annotations[train_num:], batch_size=20)

# load config
im_size = cfg.IM_SIZE

# set settings
log_dir = 'logs/000/'
weights_path = 'logs/yolo.h5'


# create model
input_shape = tuple(list(im_size)[::-1]) + (3,)
inputs = Input(shape=input_shape)
model = yolo_model(inputs, trainable=False)
if os.path.exists(weights_path):
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)


logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                             monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)


model.compile(optimizer=Adam(lr=1e-3), loss=yolo_loss)

batch_size = 32

model.fit_generator(generator=voc_data_g,
                    steps_per_epoch=max(1, train_num // batch_size),
                    validation_data=val_data_g,
                    validation_steps=max(1, val_num // batch_size),
                    epochs=50,
                    initial_epoch=0,
                    callbacks=[logging, checkpoint, reduce_lr, early_stopping])

model.save_weights(log_dir + 'trained_weights_stage_1.h5')
