from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


class DataLoader:

    def __init__(self):
        self.ds_train = None
        self.ds_valid = None
        self.ds_test = None
        self.len_train = None
        self.len_valid = None
        self.len_test = None

    def get_train_data(self, configs):
        train_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.ds_train = train_data_gen.flow_from_directory(configs['training']['filepath'],
                                                           class_mode=configs['data']['class_mode'],
                                                           target_size=configs['data']['target_size'],
                                                           batch_size=configs['data']['batch_size'],
                                                           shuffle=configs['data']['shuffle'],
                                                           )

        return self.ds_train

    def get_validation_data(self, configs):
        valid_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.ds_valid = valid_data_gen.flow_from_directory(configs['validation']['filepath'],
                                                           class_mode=configs['data']['class_mode'],
                                                           target_size=configs['data']['target_size'],
                                                           batch_size=configs['data']['batch_size'],
                                                           )

        return self.ds_valid

    def get_test_data(self, configs):
        test_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.ds_test = test_data_gen.flow_from_directory(configs['test']['filepath'],
                                                         target_size=configs['data']['target_size'],
                                                         batch_size=configs['data']['batch_size'],
                                                         )

        return self.ds_test

    def get_train_length(self):
        return self.ds_train.n

    def get_validation_length(self):
        return self.ds_valid.n

    def get_test_length(self):
        return self.ds_test.n
