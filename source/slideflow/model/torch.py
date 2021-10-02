import os
import types
import time
import torch
import torchvision
import numpy as np
import slideflow as sf
from slideflow.io.torch import interleave_dataloader
from tqdm import tqdm
from vit_pytorch import ViT
from slideflow.util import log, StainNormalizer
from slideflow.model import base as _base
import slideflow.statistics
import logging

logging.getLogger('slideflow').setLevel(logging.DEBUG)
#import slideflow.model.base as _base

def ViTmodel(image_size, num_classes):
    return ViT(
            image_size = image_size,
            patch_size = 32,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

class HyperParameters(_base.HyperParameters):
    """Build a set of hyperparameters."""

    OptDict = {
        'Adam': torch.optim.Adam
    }
    ModelDict = {
        'ViT': ViTmodel,
        'resnet18': torchvision.models.resnet18,
        'resnet50': torchvision.models.resnet50,
        'resnext50_32x4d': torchvision.models.resnext50_32x4d,
        'vgg16': torchvision.models.vgg16, # needs support added
        'mobilenet_v3_small': torchvision.models.mobilenet_v3_small
    }
    LossDict = {
        'sparse_categorical_crossentropy': torch.nn.CrossEntropyLoss
    }

    _AllLoss = ['sparse_categorical_crossentropy']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.model in self.ModelDict.keys()
        assert self.optimizer in self.OptDict.keys()
        assert self.loss in self.LossDict

    def get_opt(self):
        pass

    def get_model(self, num_classes, pretrained=None):
        if self.model in ('ViT',):
            return self.ModelDict[self.model](image_size=self.tile_px, num_classes=num_classes)
        else:
            _model = self.ModelDict[self.model](pretrained=pretrained)
            if self.model in ('resnet18', 'resnet50', 'resnext50_32x4d'):
                num_ftrs = _model.fc.in_features
                _model.fc = torch.nn.Linear(num_ftrs, num_classes)
            return _model

    def model_type(self):
        return 'categorical'

class Trainer(_base.Trainer):
    def __init__(self, hp, outdir, labels, patients, name=None, manifest=None, normalizer=None, normalizer_source=None):
        self.hp = hp
        self.outdir = outdir
        self.labels = labels
        self.patients = patients
        self.name = name
        self.manifest = manifest
        self.normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)
        self._build_model(num_classes=2, pretrain=None, checkpoint=None)

    def _build_model(self, num_classes, pretrain=None, checkpoint=None):
        self.model = self.hp.get_model(num_classes)

    def load_checkpoint(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def train(self, train_tfrecords, val_tfrecords, validate_on_batch=512, validation_batch_size=32, validation_steps=100,
              max_tiles_per_slide=0, min_tiles_per_slide=0, save_predictions=None, skip_metrics=None, seed=0,
              pretrain=None, resume_training=None, checkpoint=None, log_frequency=None, starting_epoch=None,
              ema_observations=None, ema_smoothing=None, steps_per_epoch_override=None, use_tensorboard=None,
              multi_gpu=None):

        # Temporary arguments; to be replaced once in a multi-GPU training loop
        rank = 0
        num_gpus = 1

        # Print hyperparameters
        log.info(f'Hyperparameters: {self.hp}')

        # Not implemented errors
        if pretrain is not None:
            raise NotImplementedError
        if resume_training is not None:
            raise NotImplementedError
        if checkpoint is not None:
            raise NotImplementedError
        if log_frequency is not None:
            raise NotImplementedError
        if starting_epoch is not None:
            raise NotImplementedError
        if ema_observations is not None:
            raise NotImplementedError
        if ema_smoothing is not None:
            raise NotImplementedError
        if steps_per_epoch_override is not None:
            raise NotImplementedError
        if use_tensorboard is not None:
            raise NotImplementedError
        if multi_gpu is not None:
            raise NotImplementedError
        if save_predictions is not None:
            raise NotImplementedError
        if skip_metrics is not None:
            raise NotImplementedError

        self._save_manifest(train_tfrecords, val_tfrecords)

        device = torch.device("cuda:0")

        # Setup dataloaders
        interleave_args = types.SimpleNamespace(
            tile_px=self.hp.tile_px,
            infinite=False,
            rank=rank,
            num_replicas=num_gpus,
            labels=self.labels,
            seed=seed,
            chunk_size=16,
            normalizer=self.normalizer,
            balance='none',
            manifest=self.manifest,
            max_tiles=max_tiles_per_slide,
            min_tiles=min_tiles_per_slide,
            pin_memory=True,
            num_workers=8,
        )
        dataloaders = {
            'train': interleave_dataloader(train_tfrecords, batch_size=self.hp.batch_size, augment=True, **vars(interleave_args)),
        }
        if val_tfrecords:
            dataloaders['val'] = interleave_dataloader(val_tfrecords, batch_size=validation_batch_size, augment=False, incl_slidenames=True, **vars(interleave_args))
            val_log_msg = '' if not validate_on_batch else f'every {str(validate_on_batch)} steps and '
            log.debug(f'Validation during training: {val_log_msg}at epoch end')
            if validation_steps:
                num_samples = validation_steps * self.hp.batch_size
                log.debug(f'Using {validation_steps} batches ({num_samples} samples) each validation check')
            else:
                log.debug(f'Using entire validation set each validation check')
        else:
            log.debug('Validation during training: None')

        #device = torch.device("cuda:0")
        self.model = self.model.to(device)
        params_to_update = self.model.parameters()
        optimizer = torch.optim.Adam(params_to_update, lr=0.0001)
        loss_fn = torch.nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()

        self.hp.epochs=[1]

        # Training loop
        for epoch in range(max(self.hp.epochs)):
            log.info(sf.util.bold('Epoch ' + str(epoch) + '/' + str(max(self.hp.epochs)-1)))

            for phase in dataloaders:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                num_records = 0
                running_loss = 0.0
                running_corrects = 0
                starttime = time.time()
                phase_batch_size = self.hp.batch_size if phase == 'train' else validation_batch_size

                # Setup up mid-training validation
                mid_train_val_dts = iter(dataloaders['val']) if (phase == 'train' and val_tfrecords) else None

                dataloader_pb = tqdm(total=dataloaders[phase].num_tiles,
                                     ncols=100,
                                     unit='img',
                                     leave=False)

                for i, record in enumerate(dataloaders[phase]):
                    if phase == 'train':
                        images, labels = record
                    else:
                        images, labels, _ = record

                    num_records += phase_batch_size
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss = loss_fn(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            #loss.backward()
                            #optimizer.step()

                    # Mid-training validation
                    if mid_train_val_dts is not None and (i % validate_on_batch == 0) and i > 0:
                        running_val_loss = 0
                        running_val_correct = 0
                        num_val = 0
                        for v, (val_img, val_label, _) in enumerate(mid_train_val_dts):
                            val_img = val_img.to(device)
                            val_label = val_label.to(device)
                            if validation_steps and v > validation_steps: break
                            val_outputs = self.model(val_img)
                            _, val_preds = torch.max(val_outputs, 1)
                            running_val_loss += loss.item() * val_img.size(0)
                            running_val_correct += torch.sum(val_preds == val_label.data)
                            num_val += validation_batch_size
                        val_loss = running_val_loss / num_val
                        val_acc = running_val_correct / num_val
                        log.info(f'Batch {i}: val loss: {val_loss:.4f} val acc: {val_acc:.4f}')

                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    dataloader_pb.set_description(f'{phase} loss: {running_loss / num_records:.4f} acc: {running_corrects / num_records:.4f}')
                    dataloader_pb.update(dataloaders[phase].batch_size)

                elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - starttime))
                epoch_loss = running_loss / num_records
                epoch_acc = running_corrects.double() / num_records
                log.info(f'{phase} Epoch {epoch} | loss: {epoch_loss:.4f} acc: {epoch_acc:.4f} (Elapsed: {elapsed})')
                save_path = os.path.join(self.outdir, f'saved_model_epoch{epoch}')
                if epoch in self.hp.epochs:
                    torch.save(self.model.state_dict(), save_path)
                log.info(f"Model saved to {sf.util.green(save_path)}")

                # Calculate metrics
                if val_tfrecords:
                    metrics = sf.statistics.metrics_from_dataset(self.model,
                                                                model_type='categorical',
                                                                labels=self.labels,
                                                                patients=self.patients,
                                                                manifest=self.manifest,
                                                                dataset=dataloaders['val'],
                                                                data_dir=self.outdir,
                                                                method='torch')
                    print(metrics)

class LinearTrainer(Trainer):
    def __init__(self):
        raise NotImplementedError

class CPHTrainer(Trainer):
    def __init__(self):
        raise NotImplementedError

def test_train():
    sf_config   = {'tile_px':512,
                   'tile_um':302,
                   'project_path':'/mnt/data/projects/TCGA_HNSC_1.11_TEST',
                   'outcome_label_headers':'MATH_BINARY_46',
                   'model_type':'categorical',
                   'normalizer':None,
                   'filters': {'MATH_BINARY_46': ['HIGH', 'LOW'], 'HPV_status': ['negative']}}

    #sf_config   = {'tile_px':299, 'tile_um':302, 'project_path':'/mnt/data/projects/TCGA_THCA_BRAF', 'outcome_label_headers':'brs_class', 'model_type':'categorical'}

    SFP = sf.Project(sf_config['project_path'])
    dataset = SFP.get_dataset(sf_config['tile_px'], sf_config['tile_um'], filters=sf_config['filters'])
    #dataset = SFP.get_dataset(299, 302, filter_blank=['brs_class'])
    labels, _ = dataset.labels(sf_config['outcome_label_headers'])
    training_tfrecords, val_tfrecords = dataset.training_validation_split('categorical', labels=labels, val_strategy='k-fold', val_k_fold=3, k_fold_iter=1)

    hp = HyperParameters(tile_px=sf_config['tile_px'], batch_size=128, epochs=[1,3,5,10], model='resnet18')

    trainer = Trainer(hp, '/mnt/data/tmp/vit', labels=labels, patients=dataset.patients(), manifest=dataset.manifest())
    trainer.train(training_tfrecords, val_tfrecords)