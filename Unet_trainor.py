from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import sys
from os.path import join
from Unet_box.EOS_tools import data_creator
from Unet_box.Unet import UNET

def model_train(X_train, Y_train, X_test, Y_test, model_name:str=None, model_p:str=None, dst:str='Unet_box'):
    if not model_name:
        model_name = 'model-eos3' 
    model = UNET()
    if model_p:
        model.load_weights(model_p)

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint(join(dst, model_name+'.h5'), verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, batch_size=18,
            verbose=1, epochs=100,
            validation_data=(X_test, Y_test),
            callbacks = [earlystopper, checkpointer,
                        TensorBoard(log_dir=join(dst, model_name+'_log'), histogram_freq=0)])
    return model

def training_from_dir(dir_train, dir_test, model_name=None, model_p=None):
    if dir_test == 'No':
        X, Y = data_creator(join(dir_train, 'raw_imgs'), join(dir_train, 'labels'))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    else:    
        X_train, Y_train = data_creator(join(dir_train, 'raw_imgs'), join(dir_train, 'labels'))
        X_test, Y_test = data_creator(join(dir_test, 'raw_imgs'), join(dir_test, 'labels'))
    trained_model = model_train(X_train, Y_train, X_test, Y_test, model_name, model_p)
    return trained_model
    
if __name__ == "__main__":
    dir_train = sys.argv[1]
    dir_test = sys.argv[2]
    model_name, model_p = None, None
    try:
        model_name = sys.argv[3]
    except IndexError:
        print('Use defalut name: model-eos3.h5')
    try:
        model_p = sys.argv[4]
    except IndexError:
        print('Train model from scratch')

    _ = training_from_dir(dir_train, dir_test, model_name, model_p)