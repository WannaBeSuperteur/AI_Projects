
import torch
from train_cnn import load_cnn_model_before_train, load_dataset, train_cnn, test_cnn, convert_to_train_data
from train_cnn import NUM_CLASSES, EMBEDDING_DIM_COUNT_FOR_HPO_TRAIN_DATA

import random
import numpy as np
import pandas as pd

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from hidden_representation.auto_encoder import AutoEncoderEncoder_1_28_28, AutoEncoderEncoder_3_32_32


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
TRIALS_PER_DATASET = 400


def initialize_data_dict():
    data_dict = {'total_train_images': [],
                 'max_min_of_labels': [],
                 'avg_std_of_labels': [],
                 'largest_label_percentage': [],
                 'hp_dropout_conv_earlier': [],
                 'hp_dropout_conv_later': [],
                 'hp_dropout_fc': [],
                 'hp_lr': [],
                 'hp_activation_func': [],
                 'hp_optimizer': [],
                 'hp_scheduler': []}

    for i in range(EMBEDDING_DIM_COUNT_FOR_HPO_TRAIN_DATA):
        data_dict[f'encoding_mean_{i}'] = []
        data_dict[f'encoding_std_{i}'] = []

    for i in range(NUM_CLASSES):
        data_dict[f'labels_trained_{i}'] = []

    data_dict['f1_score_macro'] = []
    return data_dict


def initialize_hp_candidates():
    return {'dropout_conv_earlier': list(np.linspace(0.0, 0.3, 301)),
            'dropout_conv_later': list(np.linspace(0.0, 0.3, 301)),
            'dropout_fc': list(np.linspace(0.0, 0.6, 601)),
            'lr': [0.00002, 0.000025, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.000085, 0.0001, 0.000125,
                   0.00015, 0.000175, 0.0002, 0.00025, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.00085,
                   0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006],
            'activation_func': ['relu', 'leaky_relu'],
            'optimizer': ['adam', 'adamw'],
            'scheduler': ['exp_90', 'exp_95', 'exp_98', 'cosine']}


def generate_constraints(dataset_name):
    if dataset_name == 'cifar_10':
        hue_left, hue_range_size = random.randint(0, 64), random.randint(16, 128)
        sat_left, sat_range_size = random.randint(0, 96), random.randint(16, 96)
        val_left, val_range_size = random.randint(96, 192), random.randint(16, 128)
        std_r_left, std_r_range_size = random.randint(0, 64), random.randint(16, 48)
        std_g_left, std_g_range_size = random.randint(0, 64), random.randint(16, 48)
        std_b_left, std_b_range_size = random.randint(0, 64), random.randint(16, 48)

        constraints = {'hue': [hue_left, hue_left + hue_range_size],
                       'saturation': [sat_left, sat_left + sat_range_size],
                       'value': [val_left, val_left + val_range_size],
                       'std_r': [std_r_left, std_r_left + std_r_range_size],
                       'std_g': [std_g_left, std_g_left + std_g_range_size],
                       'std_b': [std_b_left, std_b_left + std_b_range_size]}

    elif dataset_name == 'fashion_mnist':
        val_left, val_range_size = random.randint(0, 144), random.randint(1, 80)
        std_left, std_range_size = random.randint(0, 128), random.randint(1, 48)

        constraints = {'value': [val_left, val_left + val_range_size],
                       'std': [std_left, std_left + std_range_size]}

    else:  # mnist
        val_left, val_range_size = random.randint(0, 64), random.randint(1, 48)
        std_left, std_range_size = random.randint(32, 128), random.randint(1, 32)

        constraints = {'value': [val_left, val_left + val_range_size],
                       'std': [std_left, std_left + std_range_size]}

    return constraints


# 학습 데이터 DataFrame 을 만들기 위한 dict 에 입력/출력값 반영
# Create Date: 2026.03.20
# Last Update Date: -

# Arguments:
# - data_dict             (dict) : 학습 데이터 DataFrame 을 만들기 위한 dict
# - hpo_model_input_data  (dict) : HPO 모델의 학습 데이터 입력값의 dict
# - hpo_model_output_data (dict) : HPO 모델의 학습 데이터 출력값의 dict

def add_train_data(data_dict, hpo_model_input_data, hpo_model_output_data):

    # input data
    for i in range(EMBEDDING_DIM_COUNT_FOR_HPO_TRAIN_DATA):
        data_dict[f'encoding_mean_{i}'].append(round(hpo_model_input_data['encoding_mean'][i], 3))
        data_dict[f'encoding_std_{i}'].append(round(hpo_model_input_data['encoding_std'][i], 3))

    data_dict['total_train_images'].append(hpo_model_input_data['total_train_images'])
    data_dict['max_min_of_labels'].append(round(hpo_model_input_data['max_min_of_labels'], 3))
    data_dict['avg_std_of_labels'].append(round(hpo_model_input_data['avg_std_of_labels'], 3))
    data_dict['largest_label_percentage'].append(round(hpo_model_input_data['largest_label_percentage'], 4))

    for i in range(len(hpo_model_input_data['labels_trained'])):
        data_dict[f'labels_trained_{i}'].append(hpo_model_input_data['labels_trained'][i])

    data_dict['hp_dropout_conv_earlier'].append(round(hpo_model_input_data['hp_dropout_conv_earlier'], 4))
    data_dict['hp_dropout_conv_later'].append(round(hpo_model_input_data['hp_dropout_conv_later'], 4))
    data_dict['hp_dropout_fc'].append(round(hpo_model_input_data['hp_dropout_fc'], 4))
    data_dict['hp_lr'].append(hpo_model_input_data['hp_lr'])
    data_dict['hp_activation_func'].append(hpo_model_input_data['hp_activation_func'])
    data_dict['hp_optimizer'].append(hpo_model_input_data['hp_optimizer'])
    data_dict['hp_scheduler'].append(hpo_model_input_data['hp_scheduler'])

    # output data
    data_dict['f1_score_macro'].append(hpo_model_output_data['f1_score_macro'])


if __name__ == '__main__':
    dataset_names = ['cifar_10', 'fashion_mnist', 'mnist']

    # load Auto-Encoder encoder models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae_encoders = {}

    for dataset_name in dataset_names:
        if dataset_name == 'cifar_10':
            ae_encoder = AutoEncoderEncoder_3_32_32()
        else:
            ae_encoder = AutoEncoderEncoder_1_28_28()

        model_path = f'{PROJECT_DIR_PATH}/models/ae_encoder_{dataset_name}.pt'
        ae_encoder_state_dict = torch.load(model_path, map_location=device, weights_only=True)
        ae_encoder.load_state_dict(ae_encoder_state_dict)
        ae_encoder.to(device)
        ae_encoder.device = device
        ae_encoders[dataset_name] = ae_encoder

    # hyper-params configuration
    hp_candidates = initialize_hp_candidates()

    # demo test
    for dataset_name in dataset_names:
        print(f'\n==== DATASET: {dataset_name} ====\n')

        # initialize dict
        data_dict = initialize_data_dict()
        data_csv_path = f'{PROJECT_DIR_PATH}/hpo_training_data/test/{dataset_name}/hpo_model_train_dataset_df_10.csv'

        # hyper params
        current_trial = 0

        while current_trial < TRIALS_PER_DATASET:
            hps = {'dropout_conv_earlier': random.choice(hp_candidates['dropout_conv_earlier']),
                   'dropout_conv_later': random.choice(hp_candidates['dropout_conv_later']),
                   'dropout_fc': random.choice(hp_candidates['dropout_fc']),
                   'lr': random.choice(hp_candidates['lr']),
                   'activation_func': random.choice(hp_candidates['activation_func']),
                   'optimizer': random.choice(hp_candidates['optimizer']),
                   'scheduler': random.choice(hp_candidates['scheduler'])}
            constraints = generate_constraints(dataset_name)

            cnn_model = load_cnn_model_before_train(dataset_name, hps)
            train_dataset, valid_dataset, test_dataset, train_dataset_label_distrib, test_dataset_label_distrib, labels_trained =(
                load_dataset(dataset_name, constraints))

            # check label distribution condition
            too_many_train_data = sum(train_dataset_label_distrib) > 1500
            too_few_classes = len(train_dataset_label_distrib) < 5 or len(test_dataset_label_distrib) < 5
            too_few_minor_class_data = too_few_classes or train_dataset_label_distrib[4] < 125 or test_dataset_label_distrib[4] < 25

            if too_many_train_data or too_few_minor_class_data:
                print(f'rejected distribution: train={train_dataset_label_distrib}, test={test_dataset_label_distrib}')
                continue

            current_trial += 1

            # train Base CNN using hyper-params
            val_loss_list, best_epoch_model = train_cnn(cnn_model, train_dataset, valid_dataset)
            accuracy, f1_score_macro, f1_score_micro = test_cnn(best_epoch_model, test_dataset)
            print(f'accuracy : {accuracy}, f1_score: (macro: {f1_score_macro}, micro: {f1_score_micro})')

            # convert to HPO model training data
            ae_encoder = ae_encoders[dataset_name]
            hpo_model_input_data, hpo_model_output_data = (
                convert_to_train_data(ae_encoder, train_dataset, hps, train_dataset_label_distrib, labels_trained,
                                      f1_score_macro))

            add_train_data(data_dict, hpo_model_input_data, hpo_model_output_data)
            data_df = pd.DataFrame(data_dict)
            data_df.to_csv(data_csv_path, index=False)
