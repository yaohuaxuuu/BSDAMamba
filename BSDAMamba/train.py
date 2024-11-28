import os
import sys
import json
import PIL
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tensorboardX import SummaryWriter
from tqdm import tqdm
import logging
from datasets import get_dataset_and_info
from networks import get_model
from MedMamba import VSSM as medmamba  # import model
from bsda_warp import BSDAWarp, BSDALayer
from base import get_base_transform
from cutmix import CutMix
from tools import AverageMeter
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report,f1_score, recall_score,confusion_matrix

from __init__ import get_model


class EarlyStopping:
    def __init__(self, monitor='val_loss', min_delta=0.003, patience=50, verbose=1, mode='auto'):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.best = None
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

        if mode not in ['auto', 'min', 'max']:
            raise ValueError("mode {} is unknown!".format(mode))

        if mode == 'min':
            self.monitor_op = lambda a, b: a < b - self.min_delta
            self.best = float('inf')
        elif mode == 'max':
            self.monitor_op = lambda a, b: a > b + self.min_delta
            self.best = -float('inf')
        else:  # auto
            if 'acc' in self.monitor:
                self.monitor_op = lambda a, b: a > b + self.min_delta
                self.best = -float('inf')
            else:
                self.monitor_op = lambda a, b: a < b - self.min_delta
                self.best = float('inf')

    def on_epoch_end(self, epoch, current):
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
            self.wait += 1

    def on_train_end(self):
        if self.verbose and self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


def get_logger(results_save_path: str):
    logger = logging.getLogger(__name__)
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # # file_handler = logging.FileHandler(os.path.join(results_save_path, 'log.log'))
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)
    # 添加FileHandler到Logger对象
    # logger.addHandler(file_handler)
    return logger


def train(model, train_loader, criterion, optimizer, logger, writer, epoch, task='', alpha=0.5):
    model.train()
    total_loss = []
    ratio = min(alpha * (epoch / (100 // 2)), alpha)
    batch_time = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader),total=len(train_loader), desc='Train',mininterval=0.5):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, is_train=True)
        # print(f'train_outputs: {outputs}')
        # loss = model.get_loss(outputs, targets, criterion, logger, writer, epoch, is_train=True, bsda_alpha=ratio,
        #                       task=task)
        loss = model.get_loss(outputs, targets, criterion, logger, writer, epoch, is_train=True, 
                              task=task)

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        batch_time.update(time.time() - end)
        end = time.time()

    epoch_loss = np.mean(total_loss)
    return epoch_loss, batch_time.sum


def test(model, test_loader, criterion, device, logger, writer, split, task):
    model.eval()
    total_loss = []
    y_scores = torch.tensor([]).to(device)
    y_s = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader),total=len(test_loader), desc='Testing',mininterval=1):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # print(f'outputs: {outputs.shape}')
            loss = model.get_loss(outputs, targets, criterion, logger, writer, epoch, task=task)
            if task == 'multi-label, binary-class':
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)

            total_loss.append(loss.item())
            y_scores = torch.cat((y_scores, outputs), dim=0)
            y_s = torch.cat((y_s, targets), dim=0)

        y_s = y_s.detach().cpu().numpy()
        y_scores = y_scores.detach().cpu().numpy()
        # if y_scores.shape[1] > 2:
        #     auc, acc = roc_auc_score(y_s, y_scores, multi_class='ovr', average='weighted'), accuracy_score(y_s,
        #                                                                                                    np.argmax(
        #                                                                                                        y_scores,
        #                                                                                                        axis=1))
        # else:
        #     auc, acc = roc_auc_score(y_s, y_scores[:, 1]), accuracy_score(y_s, np.argmax(y_scores, axis=1))

        if y_scores.shape[1] > 2:
            cm = confusion_matrix(y_s, np.argmax(y_scores, axis=1))
            tn, fp, fn, tp = cm.sum(axis=0)[0], cm.sum(axis=1)[1:], cm.sum(axis=0)[-1], cm[1, 1]
            tn_fp = tn + fp  # 所有类别的真负例和假正例总和
            tn_fn = tn + fn  # 所有类别的真负例和假负例总和
            if tn_fp.sum() > 0:  # 避免除以零
                specificity = tn / (tn_fp.sum())
            else:
                specificity = 0
            f1 = f1_score(y_s, np.argmax(y_scores, axis=1), average='weighted')
            auc, acc,f1 = roc_auc_score(y_s, y_scores, multi_class='ovr', average='weighted'), accuracy_score(y_s,
                                                                                                           np.argmax(
                                                                                                               y_scores,
                                                                                                               axis=1)),f1_score(y_s, np.argmax(y_scores, axis=1), average='weighted')
            # Calculate Sensitivity and F1-score
            # For multi-class classification
            # sensitivity = recall_score(y_s, np.argmax(y_scores, axis=1), average='weighted')
            # f1 = f1_score(y_s, np.argmax(y_scores, axis=1), average='weighted')
        else:
            cm = confusion_matrix(y_s, np.argmax(y_scores, axis=1))
            tn, fp, fn, tp = cm.ravel()
            if (tn + fp) > 0:  # 避免除以零
                specificity = tn / (tn + fp)
            else:
                specificity = 0
            f1 = f1_score(y_s, np.argmax(y_scores, axis=1))
            specificity = tn / (tn + fp)  # 计算特异性
            auc, acc, f1= roc_auc_score(y_s, y_scores[:, 1]), accuracy_score(y_s, np.argmax(y_scores, axis=1)),f1_score(y_s, np.argmax(y_scores, axis=1), average='weighted')
        test_loss = np.mean(total_loss)
    return test_loss, auc, acc,specificity, f1


def main():
    global device, epoch
    results_save_path = './training_results_brain.json'
    logger = get_logger(results_save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    epoch = 0

    root_dir = "/root/MedMamba-main/MEDdata/brain"
    (train_dataset, val_dataset), data_info = get_dataset_and_info("brain","CutMix",True, root_dir)
    cutmix_train_dataset = CutMix(
        dataset=train_dataset,
        num_class=len(train_dataset.classes),
        num_mix=1,  # Number of mixed samples
        beta=1.0,  # Beta parameter for the Beta distribution
        prob=1.0  # Probability of applying CutMix
    )

    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 12])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    val_num = len(val_dataset)
    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")

    data_info = {
        'n_classes': len(train_dataset.classes),
        'n_channels': 3,
        'task': 'classification',
        'size': 224,
        'bsda_lambda': 0.8,
        'bsda_multi': 10,
        'bsda_use_ori': True,
        'brsda_multi': 10,
        'brsda_use_ori': True,
        'brsda_lambda': 0.8,
        'isda_lambda': 0.5
    }
    first_image, first_label = next(iter(train_loader))
    print('进入get_model之前')
    print(first_image.shape)
    net, model_info = get_model('medmamba', False, True, data_info)


    net = net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    model_name = 'medmamba'

    epochs = 100
    # addictional_time = AverageMeter()
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)

    writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

    training_results = {
        'epoch': [],
        'train_loss': [],
        'val_accuracy': [],
        'val_auc':[],
        'specificity': [],
        'val_f1': [],
    }

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.003, patience=50, verbose=1, mode='auto')

    for epoch in range(epochs):
        if early_stopping.stop_training:
            break

        # train
        train_loss, train_time = train(net, train_loader, loss_function, optimizer, logger, writer, epochs,
                                       data_info['task'], 0.5)
        # addictional_time.update(train_time, 1)

        # validate
        # val_loss, val_auc, val_acc = test(net, validate_loader, loss_function, device, logger, writer, 'val',
        #                                   data_info['task'])
        #
        # print('[epoch %d] train_loss: %.3f  val_auc: %.3f val_acc: %.3f' %
        #       (epoch + 1, train_loss, val_auc, val_acc))

        val_loss, val_auc, val_acc, specificity, val_f1 = test(net, validate_loader, loss_function, device, logger,
                                                               writer, 'val', data_info['task'])
        print('[epoch %d] train_loss: %.3f  val_auc: %.3f val_acc: %.3f specificity: %.3f val_f1: %.3f' %
              (epoch + 1, train_loss, val_auc, val_acc, specificity, val_f1))

        early_stopping.on_epoch_end(epoch, val_loss)

        # if val_auc > best_auc:
        #     best_epoch = epoch
        #     best_auc = val_auc
        #     best_model = deepcopy(model)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)


            # 记录结果
            training_results['epoch'].append(epoch + 1)
            training_results['train_loss'].append(train_loss)
            training_results['val_accuracy'].append(val_acc)
            training_results['val_auc'].append(val_auc)
            training_results['specificity'].append(specificity)
            training_results['val_f1'].append(val_f1)

    early_stopping.on_train_end()
    writer.close()

    # 记录日志
    # 记录运行时间
    # logger.critical('total time: %.3f, avg time: %.3f' % (addictional_time.sum, addictional_time.ave))
    # train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    # val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    # test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])
    # log = '%s\n' % (args.data_flag) + train_log + val_log + test_log
    # logger.critical(log)

    # results_save_path = './training_results_covid19.json'
    with open(results_save_path, 'w') as f:
        json.dump(training_results, f, indent=4)

    print('Finished Training')
    print('Training results saved to {}'.format(results_save_path))


if __name__ == '__main__':
    main()
