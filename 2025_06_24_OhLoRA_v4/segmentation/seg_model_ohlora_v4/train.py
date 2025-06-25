
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

import os
import numpy as np
import cv2

from seg_model_ohlora_v4.model import SegModelForOhLoRAV4


PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
VALID_RESULT_PATH = f'{PROJECT_DIR_PATH}/segmentation/segmentation_results_ohlora_v4/valid'
TEST_RESULT_PATH = f'{PROJECT_DIR_PATH}/segmentation/segmentation_results_ohlora_v4/test'
os.makedirs(VALID_RESULT_PATH, exist_ok=True)
os.makedirs(TEST_RESULT_PATH, exist_ok=True)


TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
SEG_IMAGE_SIZE = 224

MAX_EPOCHS = 1
EARLY_STOPPING_ROUNDS = 10

device = None


seg_model_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])


class OhLoRAV4SegmentationModelDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = seg_model_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        mask = cv2.imread(self.mask_paths[idx])
        mask = mask[:, :, 0] / 255.0             # 0-255 -> 0-1 linear normalize
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask


# Oh-LoRA v4 용 경량화된 Segmentation Model 정의
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - model (nn.Module) : 경량화 모델

def define_segmentation_model():
    global device

    model = SegModelForOhLoRAV4()
    model.optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    model.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=model.optimizer, gamma=0.975)

    model.to(device)
    model.device = device

    return model


# Oh-LoRA v4 용 경량화된 Segmentation Model 의 데이터셋 생성
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - dataset_path (str) : image 와 mask 가 저장되어 있는 데이터셋 경로

# Returns:
# - train_dataloader (DataLoader) : 경량화 모델의 Train Data Loader
# - valid_dataloader (DataLoader) : 경량화 모델의 Valid Data Loader
# - test_dataloader  (DataLoader) : 경량화 모델의 Test Data Loader

def generate_dataset(dataset_path):
    image_paths = list(filter(lambda x: x.startswith('face'), os.listdir(dataset_path)))
    mask_paths = list(filter(lambda x: x.startswith('parsing_hair_label'), os.listdir(dataset_path)))

    image_paths = [f'{dataset_path}/{image_path}' for image_path in image_paths]
    mask_paths = [f'{dataset_path}/{mask_path}' for mask_path in mask_paths]

    # define dataset
    dataset = OhLoRAV4SegmentationModelDataset(image_paths, mask_paths)

    # define dataloader
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    valid_size = int(dataset_size * 0.1)
    test_size = dataset_size - (train_size + valid_size)

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader


# Oh-LoRA v4 용 경량화된 Segmentation Model 학습
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - model            (nn.Module)  : 경량화 모델
# - train_dataloader (DataLoader) : 경량화 모델의 Train Data Loader
# - valid_dataloader (DataLoader) : 경량화 모델의 Valid Data Loader

# Returns:
# - valid_loss_list  (list(float)) : valid loss 의 list
# - best_epoch_model (nn.Module)   : best epoch (Loss 최소) 에서의 Segmentation Model

def train_model(model, train_dataloader, valid_dataloader):
    current_epoch = 0
    min_val_loss_epoch = -1  # Loss-based Early Stopping
    min_val_loss = None
    best_epoch_model = None
    valid_loss_list = []

    while True:
        valid_loss = train_model_each_epoch(model=model,
                                            train_dataloader=train_dataloader,
                                            valid_dataloader=valid_dataloader,
                                            current_epoch=current_epoch)

        valid_loss_list.append(valid_loss)
        print(f'epoch : {current_epoch}, val_loss : {valid_loss:.6f}')

        model.scheduler.step()

        if min_val_loss is None or valid_loss < min_val_loss:
            min_val_loss = valid_loss
            min_val_loss_epoch = current_epoch

            best_epoch_model = SegModelForOhLoRAV4().to(model.device)
            best_epoch_model.load_state_dict(model.state_dict())

        if current_epoch - min_val_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

        if current_epoch + 1 >= MAX_EPOCHS:
            break

        current_epoch += 1

    return valid_loss_list, best_epoch_model


# Oh-LoRA v4 용 경량화된 Segmentation Model 학습 (각 epoch)
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - model            (nn.Module)  : 경량화 모델
# - train_dataloader (DataLoader) : 경량화 모델의 Train Data Loader
# - valid_dataloader (DataLoader) : 경량화 모델의 Valid Data Loader
# - current_epoch    (int)        : 현재 epoch No.

# Returns:
# - valid_loss (float) : validation loss

def train_model_each_epoch(model, train_dataloader, valid_dataloader, current_epoch):

    # loss function
    loss_func = nn.MSELoss(reduction='sum')  # pred, ground-truth 모두 float 이므로 BCE Loss 사용 불가

    # run train and validation
    run_train_step(model=model,
                   train_dataloader=train_dataloader,
                   loss_func=loss_func)

    valid_loss = run_valid_step(model=model,
                                valid_dataloader=valid_dataloader,
                                loss_func=loss_func,
                                current_epoch=current_epoch)

    return valid_loss


# Oh-LoRA v4 용 경량화된 Segmentation Model 학습 (PyTorch Train step)
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - model            (nn.Module)  : 경량화 모델
# - train_dataloader (DataLoader) : 경량화 모델의 Train Data Loader
# - loss_func        (func)       : Loss Function

def run_train_step(model, train_dataloader, loss_func):
    global device

    model.train()

    for idx, (images, masks) in enumerate(train_dataloader):
        images, masks = images.to(device), masks.to(device).to(torch.float32)

        # train 실시
        model.optimizer.zero_grad()
        outputs = model(images).to(torch.float32)

        loss = loss_func(outputs, masks)
        loss.backward()
        model.optimizer.step()


# Oh-LoRA v4 용 경량화된 Segmentation Model 학습 (PyTorch Valid step)
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - model            (nn.Module)  : 경량화 모델
# - valid_dataloader (DataLoader) : 경량화 모델의 Valid Data Loader
# - loss_func        (func)       : Loss Function
# - current_epoch    (int)        : 현재 epoch No.

# Returns:
# - valid_loss (float) : validation loss

def run_valid_step(model, valid_dataloader, loss_func, current_epoch):
    global device

    model.eval()

    total = 0
    valid_loss_sum = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(valid_dataloader):

            # compute valid loss
            images, masks = images.to(device), masks.to(device).to(torch.float32)
            outputs = model(images).to(torch.float32)
            valid_loss_batch = loss_func(outputs, masks) / (VALID_BATCH_SIZE * SEG_IMAGE_SIZE * SEG_IMAGE_SIZE)
            valid_loss_sum += valid_loss_batch

            # save visualization images
            current_batch_size = masks.size(0)
            total += current_batch_size

            # generate visualization image (mask by teacher seg model: blue / output of student seg model: green)
            for img_no in range(current_batch_size):
                generate_visualization_images(images, masks, outputs, current_epoch, batch_idx, img_no)

    valid_loss = valid_loss_sum / total
    return valid_loss


# valid, test 결과에 대한 visualization image 생성
# Create Date: 2025.06.25
# Last Update Date: -

# Arguments:
# - images        (Tensor) : Segmentation 대상 얼굴 이미지
# - masks         (Tensor) : Segmentation Result (soft result of Teacher Model, for Knowledge Distillation)
# - outputs       (Tensor) : Segmentation Result (of Student Model)
# - current_epoch (int)    : 현재 epoch No. (None 이면 valid, test 중 test)
# - batch_idx     (int)    : batch No.
# - img_no        (int)    : 각 batch 안에서의 image No.

def generate_visualization_images(images, masks, outputs, current_epoch, batch_idx, img_no):
    def convert_to_numpy_img(item):
        item_ = item.detach().cpu().unsqueeze(dim=0)
        item_ = np.array(item_[0])
        item_ = np.transpose(item_, (1, 2, 0))

        item_ = item_ * 0.5 + 0.5  # de-normalize
        item_ = item_ * 255.0
        item_ = item_[:, :, ::-1]
        return item_

    # generate visualization images
    image_ = convert_to_numpy_img(images[img_no])
    mask_ = masks[img_no].detach().cpu().numpy()[0, :, :] * 255.0
    mask_ = np.expand_dims(mask_, axis=2)
    output_ = outputs[img_no].detach().cpu().numpy()[0, :, :] * 255.0
    output_ = np.expand_dims(output_, axis=2)

    overlay_y_pred = np.concatenate([mask_, output_, np.zeros((SEG_IMAGE_SIZE, SEG_IMAGE_SIZE, 1))],
                                    axis=2)
    overlay_x_y_pred = 0.55 * image_ + 0.45 * overlay_y_pred

    # save image
    img_no = batch_idx * VALID_BATCH_SIZE + img_no
    if current_epoch is None:  # test
        current_epoch_viz_dir_path = TEST_RESULT_PATH
    else:  # valid
        current_epoch_viz_dir_path = f'{VALID_RESULT_PATH}/{current_epoch}'
    os.makedirs(current_epoch_viz_dir_path, exist_ok=True)

    write_img(image_, f'{current_epoch_viz_dir_path}/img_{img_no:04d}_original_x.jpg')
    write_img(overlay_y_pred, f'{current_epoch_viz_dir_path}/img_{img_no:04d}_overlay_y_pred.jpg')
    write_img(overlay_x_y_pred, f'{current_epoch_viz_dir_path}/img_{img_no:04d}_overlay_x_y_pred.jpg')


# NumPy image 로 변환된 이미지를 파일로 저장
# Create Date: 2025.06.25
# Last Update Date: -

# Arguments:
# - image     (NumPy array) : NumPy array 형태로 변환된 이미지, dim = (H, W, 3)
# - save_path (str)         : 이미지 파일로 저장할 경로

def write_img(image, save_path):
    result, image_arr = cv2.imencode(ext='.jpg',
                                     img=image,
                                     params=[cv2.IMWRITE_JPEG_QUALITY, 95])

    if result:
        with open(save_path, mode='w+b') as f:
            image_arr.tofile(f)


# Oh-LoRA v4 용 경량화된 Segmentation Model 성능 테스트
# Create Date : 2025.06.25
# Last Update Date : -

# Arguments:
# - model           (nn.Module)  : 학습된 경량화 모델
# - test_dataloader (DataLoader) : 경량화 모델의 Test Data Loader

# Returns:
# - test_result_dict (dict) : 테스트 결과 성능지표 dict
#                             {'mse': float, 'iou': float, 'dice': float, 'recall': float, 'precision': float}

def test_model(model, test_dataloader):
    global device

    model.eval()

    total = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    mse_error_sum = 0
    mse_error_func = nn.MSELoss(reduction='sum')

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_dataloader):
            images, masks = images.to(device), masks.to(device).to(torch.float32)
            outputs = model(images).to(torch.float32)

            outputs_np = outputs.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()

            # compute MSE
            mse_error_batch = mse_error_func(outputs, masks) / (VALID_BATCH_SIZE * SEG_IMAGE_SIZE * SEG_IMAGE_SIZE)
            mse_error_sum += float(mse_error_batch.detach().cpu().numpy())

            # compute tp, tn, fp, fn
            output_bin = (outputs_np >= 0.5).astype(int)
            mask_bin = (masks_np >= 0.5).astype(int)

            tp += np.sum((output_bin == 1) & (mask_bin == 1))
            tn += np.sum((output_bin == 0) & (mask_bin == 0))
            fp += np.sum((output_bin == 1) & (mask_bin == 0))
            fn += np.sum((output_bin == 0) & (mask_bin == 1))

            # save visualization images
            current_batch_size = masks.size(0)
            total += current_batch_size

            # generate visualization image (mask by teacher seg model: blue / output of student seg model: green)
            for img_no in range(current_batch_size):
                generate_visualization_images(images=images,
                                              masks=masks,
                                              outputs=outputs,
                                              current_epoch=None,
                                              batch_idx=batch_idx,
                                              img_no=img_no)

    test_result_dict = {'mse': mse_error_sum / total,
                        'iou': tp / (tp + fp + fn + 1e-8),
                        'dice': (2 * tp) / (2 * tp + fp + fn + 1e-8),
                        'recall': tp / (tp + fn + 1e-8),
                        'precision': tp / (tp + fp + 1e-8)}

    return test_result_dict


def main():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = f'{PROJECT_DIR_PATH}/segmentation/segmentation_results_facexformer'
    train_dataloader, valid_dataloader, test_dataloader = generate_dataset(dataset_path)
    model = define_segmentation_model()

    valid_loss_list, best_epoch_model = train_model(model, train_dataloader, valid_dataloader)
    test_result_dict = test_model(model, test_dataloader)
    print(f'test result: {test_result_dict}')

    model_dir_path = f'{PROJECT_DIR_PATH}/segmentation/models'
    os.makedirs(model_dir_path, exist_ok=True)
    model_save_path = f'{model_dir_path}/segmentation_model_ohlora_v4.pth'
    torch.save(best_epoch_model.state_dict(), model_save_path)
