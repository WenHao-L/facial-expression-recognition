import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from radam import RAdam
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import datetime
import traceback

from utils.metrics.segment_metrics import eval_metrics
from utils.metrics.metrics import accuracy
from utils.generals import make_batch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class Trainer(object):
    """base class for trainers"""

    def __init__(self):
        pass



class FER2013Trainer(Trainer):
    """for classification task"""

    def __init__(self, model, train_set, val_set, test_set, configs):
        super().__init__()
        print("Start trainer..")
        print(configs)

        # load config
        self._configs = configs
        self._lr = self._configs["lr"]
        self._batch_size = self._configs["batch_size"]
        self._momentum = self._configs["momentum"] #动量，他的作用是尽量保持当前梯度的变化方向
        self._weight_decay = self._configs["weight_decay"] #（权值衰减）使用的目的是防止过拟合
        self._distributed = self._configs["distributed"]
        self._num_workers = self._configs["num_workers"]
        self._device = torch.device(self._configs["device"])
        self._max_epoch_num = self._configs["max_epoch_num"]
        self._max_plateau_count = self._configs["max_plateau_count"]

        # load dataloader and model
        self._train_set = train_set
        self._val_set = val_set
        self._test_set = test_set
        self._model = model(self._configs["in_channels"],self._configs["num_classes"])


        # # 加载预训练参数
        # self._pre_state = torch.load(os.path.join("checkpoint", "resnet18_pro_test_2022May03_14.47"))
        # self._model.load_state_dict(self._pre_state["net"])



        # self._model.fc = nn.Linear(512, 7)
        # self._model.fc = nn.Linear(256, 7)
        self._model = self._model.to(self._device)


        self._train_loader = DataLoader(
            self._train_set,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True, # pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。
            shuffle=True,
        )
        self._val_loader = DataLoader(
            self._val_set,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            shuffle=False,
        )
        self._test_loader = DataLoader(
            self._test_set,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            shuffle=False,
        )

        # define loss function (criterion) and optimizer
        class_weights = [
            1.02660468,
            9.40661861,
            1.00104606,
            0.56843877,
            0.84912748,
            1.29337298,
            0.82603942,
        ]
        class_weights = torch.FloatTensor(np.array(class_weights))

        if self._configs["weighted_loss"] == 0:
            self._criterion = nn.CrossEntropyLoss().to(self._device)
        else:
            self._criterion = nn.CrossEntropyLoss(class_weights).to(self._device)

        self._optimizer = RAdam(
            params=self._model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        #学习率优化
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            patience=self._configs["plateau_patience"],   # patience个回合如果loss没有降低，则减少学习率
            min_lr=1e-6,
            verbose=True,
        )

        """ TODO set step size equal to configs
        self._scheduler = StepLR(
            self._optimizer,
            step_size=self._configs['steplr']
        )
        """

        # training info
        self._start_time = datetime.datetime.now()
        self._start_time = self._start_time.replace(microsecond=0)

        log_dir = os.path.join(
            self._configs["cwd"],
            self._configs["log_dir"],
            "{}_{}_{}".format(
                self._configs["arch"],
                self._configs["model_name"],
                self._start_time.strftime("%Y%b%d_%H.%M"),
            ),
        )
        self._writer = SummaryWriter(log_dir)
        self._train_loss_list = []
        self._train_acc_list = []
        self._val_loss_list = []
        self._val_acc_list = []
        self._best_val_loss = 1e9
        self._best_val_acc = 0
        self._best_train_loss = 1e9
        self._best_train_acc = 0
        self._test_acc = 0.0
        self._plateau_count = 0 #当耐心次数的情况大于最大耐心值时，退出训练
        self._current_epoch_num = 0

        # for checkpoints
        # really?
        # self._checkpoint_dir = os.path.join(self._configs["cwd"], "saved/checkpoints")
        # if not os.path.exists(self._checkpoint_dir):
        #     os.makedirs(self._checkpoint_dir, exist_ok=True)

        self._checkpoint_dir = os.path.join(self._configs["cwd"], self._configs["checkpoint_dir"])
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir, exist_ok=True)



        self._checkpoint_path = os.path.join(
            self._checkpoint_dir,
            "{}_{}_{}".format(
                self._configs["arch"],
                self._configs["model_name"],
                self._start_time.strftime("%Y%b%d_%H.%M"),
            ),
        )

    def _train(self):     #训练
        self._model.train()   #状态标志
        train_loss = 0.0
        train_acc = 0.0

        # 一个epoch
        for i, (images, targets) in tqdm(
            enumerate(self._train_loader), total=len(self._train_loader), leave=False
        ):    #进度条
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            bs, c, h, w = images.shape   # bs 10 1 224 224

            images = images.view(-1,c,h,w)   # bs*10 1 224 224
            # targets = torch.repeat_interleave(targets,repeats=ncrops,dim = 0)  #复制函数，目的和10倍的images对应

            # print(targets.shape)

            # compute output, measure accuracy and record loss
            outputs = self._model(images)

            # outputs = torch.squeeze(outputs) #删除维度为一的
            # print(outputs.shape) #(128,7,1,1) 上面把1给去掉
            # print(type(targets),targets.shape) #tensor (128)

            #images.shape:320 1 200 200
            #targets.shape: 320
            #output.shape : 320 7
            loss = self._criterion(outputs, targets)  #output 和 input 的维度可以不一样，具体参考交叉熵loss函数使用
            acc = accuracy(outputs, targets)[0]
            # acc = eval_metrics(targets, outputs, 2)[0]
            # print(acc)

            train_loss += loss.item()
            train_acc += acc.item()

            # compute gradient and do SGD step
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        i += 1
        self._train_loss_list.append(train_loss / i)
        self._train_acc_list.append(train_acc / i)

    def _val(self):
        self._model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for i, (images, targets) in tqdm(
                enumerate(self._val_loader), total=len(self._val_loader), leave=False
            ):
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                bs, c, h, w = images.shape

                images = images.view(-1,c,h,w)
                # targets = torch.repeat_interleave(targets,repeats=ncrops,dim = 0)

                # compute output, measure accuracy and record loss
                outputs = self._model(images)

                outputs = torch.squeeze(outputs)

                loss = self._criterion(outputs, targets)
                acc = accuracy(outputs, targets)[0]

                val_loss += loss.item()
                val_acc += acc.item()

            i += 1
            self._val_loss_list.append(val_loss / i)
            self._val_acc_list.append(val_acc / i)

    def _test_tta(self):
        self._model.eval()
        test_acc = 0.0
        print("Calc acc on private test..")
        f = open("private_test_log.txt", "w")
        with torch.no_grad():
            for i, (images, targets) in tqdm(
                enumerate(self._test_loader), total=len(self._test_loader), leave=False
            ):

                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                bs, ncrops,c, h, w = images.shape

                images = images.view(-1,c,h,w)
                targets = torch.repeat_interleave(targets,repeats=ncrops,dim = 0)

                outputs = self._model(images)
                # outputs = torch.squeeze(outputs)
                # print(outputs.shape, outputs)
                acc = accuracy(outputs, targets)[0]
                test_acc += acc.item()
                f.writelines("{}_{}\n".format(i, acc.item()))

            test_acc = test_acc / (i + 1)
        print("Accuracy on private test: {:.3f}".format(test_acc))
        f.close()
        return test_acc


    def _calc_acc_on_private_test(self):
        self._model.eval()
        test_acc = 0.0
        print("Calc acc on private test..")
        f = open("private_test_log.txt", "w")
        with torch.no_grad():
            for i, (images, targets) in tqdm(
                enumerate(self._test_loader), total=len(self._test_loader), leave=False
            ):

                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                bs, c, h, w = images.shape

                images = images.view(-1,c,h,w)
                # targets = torch.repeat_interleave(targets,repeats=ncrops,dim = 0)

                outputs = self._model(images)
                # outputs = torch.squeeze(outputs)
                # print(outputs.shape, outputs)
                acc = accuracy(outputs, targets)[0]
                test_acc += acc.item()
                f.writelines("{}_{}\n".format(i, acc.item()))

            test_acc = test_acc / (i + 1)
        print("Accuracy on private test: {:.3f}".format(test_acc))
        f.close()
        return test_acc

    def _calc_acc_on_private_test_with_tta(self):
        # 一张一张迭代
        self._model.eval()
        test_acc = 0.0
        print("Calc acc on private test with tta..")
        f = open(
            "private_test_log_{}_{}.txt".format(
                self._configs["arch"], self._configs["model_name"]
            ),
            "w",
        )

        with torch.no_grad():
            for idx in tqdm(
                range(len(self._test_set)), total=len(self._test_set), leave=False
            ):
                images, targets = self._test_set[idx]
                targets = torch.LongTensor([targets])

                images = make_batch(images)
                images = images.cuda(non_blocking=True) #1 10 1 200 200
                targets = targets.cuda(non_blocking=True)

                bs, c, h, w = images.shape

                images = images.view(-1,c,h,w) # 10 1 200 200
                # targets = torch.repeat_interleave(targets,repeats=ncrops,dim = 0)


                outputs = self._model(images)

                outputs = F.softmax(outputs, 1)

                # outputs.shape [tta_size, 7]
                outputs = torch.sum(outputs, 0)

                outputs = torch.unsqueeze(outputs, 0)
                # print(outputs.shape)
                # TODO: try with softmax first and see the change
                acc = accuracy(outputs, targets)[0]
                test_acc += acc.item()
                f.writelines("{}_{}\n".format(idx, acc.item()))

            test_acc = test_acc / (idx + 1)
        print("Accuracy on private test with tta: {:.3f}".format(test_acc))
        f.close()
        return test_acc

    def train(self):

        try: #检测停止信号
            while not self._is_stop():
                self._increase_epoch_num()
                self._train() #训练
                self._val()   #验证
                self._update_training_state()
                self._logging()
        except KeyboardInterrupt:
            traceback.print_exc()
            pass


        # training stop and text the acc
        try:
            #state = torch.load(os.path.join("checkpoint", "resnet34_test_2022Apr04_21.05"))
            state = torch.load(self._checkpoint_path)   #加载模型

            self._model.load_state_dict(state["net"])

            # if not self._test_set.is_tta():                 #测试使用tta
            #     self._test_acc = self._calc_acc_on_private_test()
            # else:
            #     self._test_acc = self._calc_acc_on_private_test_with_tta()

            self._test_acc = self._calc_acc_on_private_test()
            self._test_acc = self._calc_acc_on_private_test_with_tta()

            self._save_weights()
        except Exception as e:
            traceback.print_exc()
            pass

        consume_time = str(datetime.datetime.now() - self._start_time)
        self._writer.add_text(
            "Summary",
            "Converged after {} epochs, consume {}".format(
                self._current_epoch_num, consume_time[:-7]
            ),
        )
        self._writer.add_text(
            "Results", "Best validation accuracy: {:.3f}".format(self._best_val_acc)
        )
        self._writer.add_text(
            "Results", "Best training accuracy: {:.3f}".format(self._best_train_acc)
        )
        self._writer.add_text(
            "Results", "Private test accuracy: {:.3f}".format(self._test_acc)
        )
        self._writer.close()

    def _update_training_state(self):
        if self._val_acc_list[-1] > self._best_val_acc:
            self._save_weights()
            self._plateau_count = 0
            self._best_val_acc = self._val_acc_list[-1]
            self._best_val_loss = self._val_loss_list[-1]
            self._best_train_acc = self._train_acc_list[-1]
            self._best_train_loss = self._train_loss_list[-1]
        else:
            self._plateau_count += 1

        # self._scheduler.step(self._train_loss_list[-1])
        self._scheduler.step(100 - self._val_acc_list[-1])
        # self._scheduler.step()

    def _logging(self): #打印训练信息
        consume_time = str(datetime.datetime.now() - self._start_time)

        message = "\nE{:03d}  {:.3f}/{:.3f}/{:.3f} {:.3f}/{:.3f}/{:.3f} | p{:02d}  Time {}\n".format(
            self._current_epoch_num,
            self._train_loss_list[-1],
            self._val_loss_list[-1],
            self._best_val_loss,
            self._train_acc_list[-1],
            self._val_acc_list[-1],
            self._best_val_acc,
            self._plateau_count,
            consume_time[:-7],
        )

        self._writer.add_scalar(
            "Accuracy/Train", self._train_acc_list[-1], self._current_epoch_num
        )
        self._writer.add_scalar(
            "Accuracy/Val", self._val_acc_list[-1], self._current_epoch_num
        )
        self._writer.add_scalar(
            "Loss/Train", self._train_loss_list[-1], self._current_epoch_num
        )
        self._writer.add_scalar(
            "Loss/Val", self._val_loss_list[-1], self._current_epoch_num
        )

        print(message)

    def _is_stop(self):
        """check stop condition"""
        return (
            self._plateau_count > self._max_plateau_count
            or self._current_epoch_num > self._max_epoch_num
        )

    def _increase_epoch_num(self):
        self._current_epoch_num += 1

    def _save_weights(self, test_acc=0.0):

        state_dict = self._model.state_dict()

        state = {
            **self._configs,
            "net": state_dict,
            "best_val_loss": self._best_val_loss,
            "best_val_acc": self._best_val_acc,
            "best_train_loss": self._best_train_loss,
            "best_train_acc": self._best_train_acc,
            "train_losses": self._train_loss_list,
            "val_loss_list": self._val_loss_list,
            "train_acc_list": self._train_acc_list,
            "val_acc_list": self._val_acc_list,
            "test_acc": self._test_acc,
        }

        torch.save(state, self._checkpoint_path)


    def test(self,checkpoint_path):
        #state = torch.load(os.path.join("checkpoint", "resnet18_test_2022Apr13_08.04"))
        state = torch.load(checkpoint_path)   #加载模型
        # print(state)
        self._model.load_state_dict(state["net"],strict=False)

        # if not self._test_set.is_tta():                 #测试使用tta
        #     self._test_acc = self._calc_acc_on_private_test_with_tta()
        # else:
        #     self._test_acc = self._calc_acc_on_private_test_with_tta()
        self._test_acc = self._calc_acc_on_private_test()

        # self._test_acc = self._test_tta()

        self._save_weights()


