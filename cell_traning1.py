from sklearn.model_selection import train_test_split
from random import shuffle, seed
import math
import os
import shutil
from keras import models, layers, optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator 
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import sys
import matplotlib.pyplot as plt

def move(yes_base, no_base):
    seed = 42
    yes = os.listdir(yes_base)
    yes_train = math.floor(len(yes) * 0.8)
#     yes_test = math.floor(len(yes) * 0.9)
    no = os.listdir(no_base)
    no_train = math.floor(len(no) * 0.8)
#     no_test = math.floor(len(no) * 0.9)
    shuffle(yes)
    shuffle(no)
    root = os.path.basename(yes_base)
    for dst, (a, b), (c, d) in zip(('train', 'validation'), 
                                   ((0, yes_train), (yes_train, len(yes)),),
                                   ((0, no_train), (no_train, len(no)))):
        dst_p = os.path.join(root, 'imporve',dst)
        os.makedirs(os.path.join(dst_p, 'yes'), exist_ok=True)
        os.makedirs(os.path.join(dst_p, 'no'), exist_ok=True)
        for y in yes[a:b]:
            start_f = os.path.join(yes_base, y)
            dst_f = os.path.join(dst_p, 'yes', y)
            shutil.copy(start_f, dst_f)
        for n in no[c:d]:
            start_f = os.path.join(no_base, n)
            dst_f = os.path.join(dst_p, 'no', n)
            shutil.copy(start_f, dst_f)

def gen_generators(train_dir, validation_dir):
    test_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(96, 96),
            batch_size=32,
            class_mode='binary')

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)

    train_generator = train_datagen.flow_from_directory(
            # This is the target directory
            train_dir,
            # All images will be resized to 150x150
            target_size=(96, 96),
            batch_size=32,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')

    return validation_generator, train_generator

def gen_load_model(mp='modelH2.h5'):
    model = models.load_model(mp)
    print("Loaded model from disk")
    return model

def gen_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(96, 96, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def plot_process(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('Training and validation accuracy.png', dpi=300)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('Training and validation loss.png', dpi=300)

def main():
    improve = sys.argv[1]
    train_dir = sys.argv[2]
    validation_dir = sys.argv[3]
    if improve == '1':
        model = gen_load_model()
    else:
        model = gen_model()

    train_generator, validation_generator = gen_generators(train_dir, validation_dir)
    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(),
                metrics=['acc'])
    cheak_list = [EarlyStopping(monitor='loss', patience=10),
                ModelCheckpoint(filepath='modelcc.h5', monitor='acc', save_best_only=True),
                TensorBoard(log_dir='cell_log', histogram_freq=0)]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=300,
        callbacks=cheak_list,
        validation_data=validation_generator,
        validation_steps=50)
    plot_process(history)

if __name__ == "__main__":
    if sys.argv[1] == 'move':
        move(sys.argv[2], sys.argv[3])
    else:
        main()