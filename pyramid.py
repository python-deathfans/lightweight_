import keras
from keras.layers import Lambda, Conv2D, BatchNormalization, Dropout, SeparableConv2D, MaxPooling2D, Input, \
    GlobalMaxPooling2D, Dense, Activation
from keras.utils import to_categorical, plot_model
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10, cifar100
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import wandb
from wandb.keras import WandbCallback
import time
from keras.models import Model
import numpy as np
from keras.initializers import he_normal


class Module:

    def __init__(self):
        self.num_class = 100
        self.batch_size = 64
        self.epoch = 200
        self.x_shape = (32, 32, 3)
        self.callback_list = self.get_callback_list()

        self.model = self.build_model()

        print("开始训练:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        start = time.time()
        self.train()
        end = time.time()

        print("结束训练:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(f"共耗时{end - start}s")

    @staticmethod
    def get_callback_list():

        callback_list = [EarlyStopping(monitor='acc',  # 监控模型的验证精度
                                       patience=5),
                         ModelCheckpoint(filepath='Net2.h5',  # 目标文件的保存路径
                                         monitor='val_loss',  # 监控验证损失
                                         save_best_only=True),
                         ReduceLROnPlateau(monitor='val_loss',  # 监控模型的验证损失
                                           factor=0.1,  # 触发时将学习率乘以系数0.1
                                           patience=5),
                         WandbCallback()
                         ]

        return callback_list

    @staticmethod
    def gp_3x3(inputs, pre_layer):

        pre_layer = int(pre_layer/2)

        normalized1_1 = Lambda(lambda x: x[:, :, :, :pre_layer])(inputs)
        normalized1_2 = Lambda(lambda x: x[:, :, :, pre_layer:])(inputs)

        # 第一路
        tower_1 = Conv2D(8, (1, 1), padding='same', activation='relu')(normalized1_1)
        x = BatchNormalization()(tower_1)
        tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
        tower_1 = BatchNormalization()(tower_1)

        # 第二路
        tower_2 = Conv2D(8, (1, 1), padding='same', activation='relu')(normalized1_2)
        x = BatchNormalization()(tower_2)
        tower_2 = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
        tower_2 = BatchNormalization()(tower_2)

        output = keras.layers.concatenate([tower_1, tower_2], axis=3)

        return output

    @staticmethod
    def gp_5x5(inputs, pre_layer):

        interval = int(pre_layer/4)

        normalized1_1 = Lambda(lambda x: x[:, :, :, :interval])(inputs)
        normalized1_2 = Lambda(lambda x: x[:, :, :, interval:2*interval])(inputs)
        normalized1_3 = Lambda(lambda x: x[:, :, :, 2*interval:3*interval])(inputs)
        normalized1_4 = Lambda(lambda x: x[:, :, :, 3*interval:])(inputs)

        # 第一路
        tower_1 = Conv2D(4, (1, 1), padding='same', activation='relu')(normalized1_1)
        x = BatchNormalization()(tower_1)
        tower_1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        tower_1 = BatchNormalization()(tower_1)
        tower_1 = Conv2D(8, (3, 3), padding='same', activation='relu')(tower_1)
        tower_1 = BatchNormalization()(tower_1)

        # 第二路
        tower_2 = Conv2D(4, (1, 1), padding='same', activation='relu')(normalized1_2)
        x = BatchNormalization()(tower_2)
        tower_2 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        tower_2 = BatchNormalization()(tower_2)
        tower_2 = Conv2D(8, (3, 3), padding='same', activation='relu')(tower_2)
        tower_2 = BatchNormalization()(tower_2)

        # 第三路
        tower_3 = Conv2D(4, (1, 1), padding='same', activation='relu')(normalized1_3)
        x = BatchNormalization()(tower_3)
        tower_3 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        tower_3 = BatchNormalization()(tower_3)
        tower_3 = Conv2D(8, (3, 3), padding='same', activation='relu')(tower_3)
        tower_3 = BatchNormalization()(tower_3)

        # 第四路
        tower_4 = Conv2D(4, (1, 1), padding='same', activation='relu')(normalized1_4)
        x = BatchNormalization()(tower_4)
        tower_4 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        tower_4 = BatchNormalization()(tower_4)
        tower_4 = Conv2D(8, (3, 3), padding='same', activation='relu')(tower_4)
        tower_4 = BatchNormalization()(tower_4)

        output = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)

        return output

    def gp_block(self, inputs):
        # 生成分组卷积模块

        # 第一路
        tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)
        tower_1 = BatchNormalization()(tower_1)

        # 第二路
        tower_2 = self.gp_3x3(inputs, 128)
        tower_2 = BatchNormalization()(tower_2)

        # 第三路
        tower_3 = self.gp_5x5(inputs, 128)

        output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)

        return output

    def py_block(self, inputs):
        # 生成金字塔卷积模块

        weight_decay = 0.0005

        x = Conv2D(128, 1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        x = self.gp_block(x)
        x = Conv2D(256, 1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)

        return x

    def build_model(self):
        weight_decay = 0.0005

        inputs = Input(shape=self.x_shape)

        x = Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal())(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal())(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal())(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal())(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        x = SeparableConv2D(256, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal())(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = SeparableConv2D(256, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=he_normal())(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        x = self.py_block(x)

        # x = GlobalAveragePooling2D()(x)
        x = GlobalMaxPooling2D()(x)
        outputs = Dense(self.num_class, activation='softmax')(x)

        model_ = Model(inputs=inputs, outputs=outputs)
        # model_ = multi_gpu_model(model_, gpus=2)

        plot_model(model_, "./pyramid.png", show_shapes=True, show_layer_names=True, dpi=120)

        model_.summary()

        return model_

    @staticmethod
    def normalize(x_train, x_test):
        mean = np.mean(x_train, axis=(0, 1, 2, 3))
        std = np.std(x_train, axis=(0, 1, 2, 3))
        x_train = (x_train - mean) / (std + 1e-7)
        x_test = (x_test - mean) / (std + 1e-7)

        return x_train, x_test

    def preprocess_data(self, x_train, y_train, x_test, y_test):
        y_train = to_categorical(y_train, self.num_class)
        y_test = to_categorical(y_test, self.num_class)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # x_test /= 255.0
        # x_train /= 255.0

        # x_train, x_test = self.normalize(x_train, x_test)

        return x_train, y_train, x_test, y_test

    def train(self):

        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train, y_train, x_test, y_test = self.preprocess_data(x_train, y_train, x_test, y_test)

        # data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(x_train)

        # 模型编译
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=self.batch_size),
                                 steps_per_epoch=x_train.shape[0] // self.batch_size,
                                 epochs=self.epoch, validation_data=(x_test, y_test),
                                 callbacks=self.callback_list)

        loss, acc = self.model.evaluate(x_test, y_test, self.batch_size)

        print(f"验证集准确率是:{acc}, 损失是:{loss}")


wandb.init(project='lightweight')

model = Module()
