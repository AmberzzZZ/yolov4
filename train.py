from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from yolov4 import yolov4
from dataSequence import dataSequence
import pickle
from config import get_config


if __name__ == '__main__':

    # args
    cfg = get_config(None)
    img_size = cfg.DATA.IMG_SIZE
    batch_size = cfg.DATA.BATCH_SIZE
    n_classes = cfg.MODEL.NUM_CLASSES
    n_anchors = cfg.ANCHOR.NA
    anchors = cfg.ANCHOR.ANCHORS
    print('anchors: ', anchors)

    # data
    train_generator = dataSequence(cfg, rect=False, augment=True, mosaic=False)

    # model
    model = yolov4(input_shape=(img_size,img_size,3), n_classes=n_classes, n_anchors=n_anchors, cfg=cfg,
                   training=1, test=False)
    model.load_weights('weights/yolov4-p5.h5')
    opt = Adam(3e-4)
    model.compile(opt, loss=lambda y_true,y_pred:y_pred, metrics=[])

    # ckpt
    filepath = 'weights/yoloP5_input%d_cls%d_epoch_{epoch:02d}_loss_{loss:.3f}.h5' % (img_size, n_classes)
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='auto', save_weights_only=True, period=1)

    # train
    history = model.fit_generator(train_generator, 
                                  steps_per_epoch=200,
                                  initial_epoch=0,
                                  epochs=300,
                                  verbose=1,
                                  callbacks=[checkpoint],
                                  workers=16,
                                  use_multiprocessing=True)
    with open('history.pkl', 'wb') as f:
        pickle.dump(history.history, f)




