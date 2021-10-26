import os
import types
import time
import torch
import torchvision
import pretrainedmodels
import slideflow as sf
import numpy as np
import slideflow.statistics

from slideflow.model import base as _base
from slideflow.util import log, StainNormalizer
from slideflow.model import torch_utils
from tqdm import tqdm
from vit_pytorch import ViT

class ModelParams(_base.ModelParams):
    """Build a set of hyperparameters."""

    OptDict = {
        'Adadelta': torch.optim.Adadelta,
        'Adagrad': torch.optim.Adagrad,
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'SparseAdam': torch.optim.SparseAdam,
        'Adamax': torch.optim.Adamax,
        'ASGD': torch.optim.ASGD,
        'LBFGS': torch.optim.LBFGS,
        'RMSprop': torch.optim.RMSprop,
        'Rprop': torch.optim.Rprop,
        'SGD': torch.optim.SGD
    }
    ModelDict = {
        'ViT': ViT,
        'resnet18': torchvision.models.resnet18,
        'resnet50': torchvision.models.resnet50,
        'alexnet': torchvision.models.alexnet,
        'squeezenet': torchvision.models.squeezenet,
        'densenet': torchvision.models.densenet161,
        'inception': torchvision.models.inception_v3,
        'googlenet': torchvision.models.googlenet,
        'shufflenet': torchvision.models.shufflenet_v2_x1_0,
        'resnext50_32x4d': torchvision.models.resnext50_32x4d,
        'vgg16': torchvision.models.vgg16,                              # needs support added
        'mobilenet_v2': torchvision.models.mobilenet_v2,
        'mobilenet_v3_small': torchvision.models.mobilenet_v3_small,
        'mobilenet_v3_large': torchvision.models.mobilenet_v3_large,
        'wide_resnet50_2': torchvision.models.wide_resnet50_2,
        'mnasnet': torchvision.models.mnasnet1_0,
        'xception': pretrainedmodels.xception
    }
    LinearLossDict = {
        'L1Loss': torch.nn.L1Loss,
        'MSELoss': torch.nn.MSELoss,
        'NLLLoss': torch.nn.NLLLoss, #negative log likelihood
        'HingeEmbeddingLoss': torch.nn.HingeEmbeddingLoss,
        'SmoothL1Loss': torch.nn.SmoothL1Loss,
        'CosineEmbeddingLoss': torch.nn.CosineEmbeddingLoss,
    }

    AllLossDict = {
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
        'CTCLoss': torch.nn.CTCLoss,
        'PoissonNLLLoss': torch.nn.PoissonNLLLoss,
        'GaussianNLLLoss': torch.nn.GaussianNLLLoss,
        'KLDivLoss': torch.nn.KLDivLoss,
        'BCELoss': torch.nn.BCELoss,
        'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss,
        'MarginRankingLoss': torch.nn.MarginRankingLoss,
        'MultiLabelMarginLoss': torch.nn.MultiLabelMarginLoss,
        'HuberLoss': torch.nn.HuberLoss,
        'SoftMarginLoss': torch.nn.SoftMarginLoss,
        'MultiLabelSoftMarginLoss': torch.nn.MultiLabelSoftMarginLoss,
        'MultiMarginLoss': torch.nn.MultiMarginLoss,
        'TripletMarginLoss': torch.nn.TripletMarginLoss,
        'TripletMarginWithDistanceLoss': torch.nn.TripletMarginWithDistanceLoss,
    }

    def __init__(self, model='xception', loss='CrossEntropyLoss', **kwargs):
        super().__init__(model=model, loss=loss, **kwargs)
        assert self.model in self.ModelDict.keys()
        assert self.optimizer in self.OptDict.keys()
        assert self.loss in self.AllLossDict

    def get_opt(self, params_to_update):
        return self.OptDict[self.optimizer](params_to_update, lr=self.learning_rate)

    def build_model(self, num_classes=None, num_slide_features=0, labels=None, pretrain=None, checkpoint=None):
        assert num_classes is not None or labels is not None
        if num_classes is None:
            num_classes = self._detect_classes_from_labels(labels)
        outcomes = list(num_classes.keys())

        if len(outcomes) > 1:
            raise NotImplementedError
        else:
            num_classes = num_classes[outcomes[0]] # Forces single categorical model for now

        if self.model == 'ViT':
            return ViT(image_size=self.tile_px,
                        patch_size=32,
                        num_classes=num_classes,
                        dim=1024,
                        depth=6,
                        heads=16,
                        mlp_dim=2048,
                        dropout=0.1,
                        emb_dropout=0.1)
        elif self.model in ('xception',):
            _model = self.ModelDict[self.model](num_classes=1000, pretrained=pretrain)
            num_ftrs = _model.last_linear.in_features
            _model.last_linear = torch.nn.Linear(num_ftrs, num_classes)
            return _model
        else:
            _model = self.ModelDict[self.model](pretrained=pretrain)
            if self.model in ('resnet18', 'resnet50', 'resnext50_32x4d'):
                num_ftrs = _model.fc.in_features
                _model.fc = torch.nn.Linear(num_ftrs, num_classes)
            return _model

    def model_type(self):
        if self.loss == 'NLLLoss':
            return 'cph'
        elif self.loss in self.LinearLossDict:
            return 'linear'
        else:
            return 'categorical'

class Trainer(_base.Trainer):
    def __init__(self, hp, outdir, labels, patients, name=None, manifest=None, slide_input=None, feature_sizes=None,
                 feature_names=None, outcome_names=None, normalizer=None, normalizer_source=None, mixed_precision=True,
                 config=None, neptune_api=None, neptune_workspace=None):

        self.hp = hp
        self.outdir = outdir
        self.labels = labels
        self.patients = patients
        self.name = name
        self.manifest = manifest
        self.model = None
        self.normalizer = None if not normalizer else StainNormalizer(method=normalizer, source=normalizer_source)
        if normalizer: log.info(f'Using realtime {normalizer} normalization')
        self.mixed_precision = mixed_precision
        self.outcome_names = outcome_names
        outcome_labels = np.array(list(labels.values()))
        if len(outcome_labels.shape) == 1:
            outcome_labels = np.expand_dims(outcome_labels, axis=1)
        if not self.outcome_names:
            self.outcome_names = [f'Outcome {i}' for i in range(outcome_labels.shape[1])]
        if not os.path.exists(outdir): os.makedirs(outdir)

    def load(self, model):
        self.model = self.hp.build_model(labels=self.labels)
        self.model.load_state_dict(torch.load(model))

    def evaluate(self, dataset, batch_size=None, permutation_importance=False, histogram=False, save_predictions=False):

        # Load and initialize model
        if not self.model:
            raise sf.util.UserError("Model has not been loaded, unable to evaluate.")
        device = torch.device('cuda')
        self.model.to(device)
        self.model.eval()
        self._save_manifest(val_tfrecords=dataset.tfrecords())
        if not batch_size: batch_size = self.hp.batch_size

        # Setup dataloaders
        interleave_args = types.SimpleNamespace(
            rank=0,
            num_replicas=1,
            labels=self.labels,
            seed=0,
            chunk_size=16,
            normalizer=self.normalizer,
            pin_memory=True,
            num_workers=8,
            onehot=False,
        )

        torch_dataset = dataset.torch(infinite=False, batch_size=batch_size, augment=False, incl_slidenames=True, **vars(interleave_args))

        metric_kwargs = types.SimpleNamespace(
            dataset=torch_dataset,
            model=self.model,
            model_type=self.hp.model_type(),
            labels=self.labels,
            patients=self.patients,
            outcome_names=self.outcome_names,
            data_dir=self.outdir,
            num_tiles=dataset.num_tiles,
            label='eval'
        )

        # Generate performance metrics
        log.info('Calculating performance metrics...')
        if permutation_importance:
            drop_images = ((self.hp.tile_px == 0) or self.hp.drop_images)
            metrics = sf.statistics.permutation_feature_importance(feature_names=self.feature_names,
                                                                   feature_sizes=self.feature_sizes,
                                                                   drop_images=drop_images,
                                                                   **vars(metric_kwargs))
        else:
            metrics = sf.statistics.metrics_from_dataset(histogram=histogram,
                                                         verbose=True,
                                                         save_predictions=save_predictions,
                                                         **vars(metric_kwargs))
        results_dict = { 'eval': {} }
        for metric in metrics:
            if metrics[metric]:
                log.info(f"Tile {metric}: {metrics[metric]['tile']}")
                log.info(f"Slide {metric}: {metrics[metric]['slide']}")
                log.info(f"Patient {metric}: {metrics[metric]['patient']}")
                results_dict['eval'].update({
                    f'tile_{metric}': metrics[metric]['tile'],
                    f'slide_{metric}': metrics[metric]['slide'],
                    f'patient_{metric}': metrics[metric]['patient']
                })

        results_log = os.path.join(self.outdir, 'results_log.csv')
        sf.util.update_results_log(results_log, 'eval_model', results_dict)
        return results_dict

    def train(self, train_dts, val_dts, validate_on_batch=512, validation_batch_size=32, validation_steps=100,
              save_predictions=True, skip_metrics=False, seed=0, log_frequency=None, starting_epoch=0,
              ema_observations=None, ema_smoothing=None, use_tensorboard=None, steps_per_epoch_override=0,
              multi_gpu=None, resume_training=None, pretrain='imagenet', checkpoint=None):

        # Temporary arguments; to be replaced once in a multi-GPU training loop
        rank = 0
        num_gpus = 1
        starting_epoch = max(starting_epoch, 1)

        # Print hyperparameters
        log.info(f'Hyperparameters: {self.hp}')

        # Enable TF32 (should be enabled by default)
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow PyTorch to internally use tf32 for matmul
        torch.backends.cudnn.allow_tf32 = True        # Allow PyTorch to internally use tf32 for convolutions

        # Not implemented errors
        if log_frequency is not None:
            raise NotImplementedError
        if ema_observations is not None:
            raise NotImplementedError
        if ema_smoothing is not None:
            raise NotImplementedError
        if use_tensorboard is not None:
            raise NotImplementedError
        if multi_gpu:
            raise NotImplementedError

        # Training preparation
        self._save_manifest(train_dts.tfrecords(), val_dts.tfrecords())
        device = torch.device("cuda")
        if steps_per_epoch_override:
            steps_per_epoch = steps_per_epoch_override
            log.info(f"Overriding steps per epoch = {steps_per_epoch_override}")
        else:
            steps_per_epoch = train_dts.num_tiles // self.hp.batch_size
            log.info(f"Steps per epoch = {steps_per_epoch}")

        # Build model
        self.model = self.hp.build_model(labels=self.labels, pretrain=pretrain)
        self.model = self.model.to(device)
        img_batch = torch.empty([self.hp.batch_size, 3, train_dts.tile_px, train_dts.tile_px], device=device)
        torch_utils.print_module_summary(self.model, [img_batch])

        # Setup dataloaders
        interleave_args = types.SimpleNamespace(
            rank=rank,
            num_replicas=num_gpus,
            labels=self.labels,
            seed=seed,
            chunk_size=16,
            normalizer=self.normalizer,
            pin_memory=True,
            num_workers=4,
            onehot=False,
        )

        dataloaders = {
            'train': iter(train_dts.torch(infinite=True, batch_size=self.hp.batch_size, augment=True, **vars(interleave_args)))
        }
        if val_dts is not None:
            dataloaders['val'] = val_dts.torch(infinite=False, batch_size=validation_batch_size, augment=False, incl_slidenames=True, **vars(interleave_args))
            val_log_msg = '' if not validate_on_batch else f'every {str(validate_on_batch)} steps and '
            log.debug(f'Validation during training: {val_log_msg}at epoch end')
            if validation_steps:
                num_samples = validation_steps * self.hp.batch_size
                log.debug(f'Using {validation_steps} batches ({num_samples} samples) each validation check')
            else:
                log.debug(f'Using entire validation set each validation check')
        else:
            log.debug('Validation during training: None')

        params_to_update = self.model.parameters()
        optimizer = self.hp.get_opt(params_to_update)
        loss_fn = torch.nn.CrossEntropyLoss()
        if self.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()

        # Training loop
        for epoch in range(starting_epoch, max(self.hp.epochs)+1):
            print()
            log.info(sf.util.bold('Epoch ' + str(epoch) + '/' + str(max(self.hp.epochs))))

            for phase in dataloaders:
                num_records = 0
                running_loss = 0.0
                running_corrects = 0
                step = 1
                starttime = time.time()

                if phase == 'train':
                    self.model.train()

                    # Setup up mid-training validation
                    mid_train_val_dts = dataloaders['val'] if (phase == 'train' and val_dts) else None

                    num_steps = steps_per_epoch * self.hp.batch_size
                    dataloader_pb = tqdm(total=num_steps, ncols=100, unit='img', leave=False)

                    while step < steps_per_epoch:
                        images, labels = next(dataloaders['train'])
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)

                        # Training step
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(True):
                            with torch.cuda.amp.autocast() if self.mixed_precision else sf.model.utils.no_scope():
                                outputs = self.model(images)
                                loss = loss_fn(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                            if self.mixed_precision:
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                optimizer.step()

                        # Update losses
                        running_loss += loss.item() * images.size(0)
                        num_correct = torch.sum(preds == labels.data)
                        running_corrects += num_correct
                        num_records = step * self.hp.batch_size
                        dataloader_pb.set_description(f'{phase} loss: {running_loss / num_records:.4f} acc: {running_corrects / num_records:.4f}')
                        dataloader_pb.update(self.hp.batch_size)

                        # Mid-training validation
                        if mid_train_val_dts is not None and (step % validate_on_batch == 0) and step > 0:
                            self.model.eval()
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
                            log.info(f'Batch {step}: val loss: {val_loss:.4f} val acc: {val_acc:.4f}')
                            self.model.train()

                        step += 1
                    dataloader_pb.close()

                # Perform basic evaluation at every epoch end
                if phase == 'val' and (val_dts is not None):
                    self.model.eval()
                    dataloader_pb = tqdm(total=dataloaders['val'].num_tiles, ncols=100, unit='img', leave=False)
                    for images, labels, slides in dataloaders['val']:
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        optimizer.zero_grad()

                        with torch.no_grad():
                            with torch.cuda.amp.autocast() if self.mixed_precision else sf.model.utils.no_scope():
                                outputs = self.model(images)
                                loss = loss_fn(outputs, labels)
                            _, preds = torch.max(outputs, 1)

                        running_loss += loss.item() * images.size(0)
                        num_correct = torch.sum(preds == labels.data)
                        running_corrects += num_correct
                        num_records = (step+1) * dataloaders['val'].batch_size
                        dataloader_pb.set_description(f'{phase} loss: {running_loss / num_records:.4f} acc: {running_corrects / num_records:.4f}')
                        dataloader_pb.update(dataloaders['val'].batch_size)
                        step += 1
                    dataloader_pb.close()

                elapsed = time.strftime('%H:%M:%S', time.gmtime(time.time() - starttime))
                epoch_loss = running_loss / num_records
                epoch_acc = running_corrects.double() / num_records
                log.info(f'{phase} Epoch {epoch} | loss: {epoch_loss:.4f} acc: {epoch_acc:.4f} (Elapsed: {elapsed})')

                # Perform full metrics if the epoch is one of the predetermined epochs at which to save/eval a model
                if phase == 'val' and (val_dts is not None) and epoch in self.hp.epochs:
                    # Calculate full evaluation metrics
                    self.model.eval()
                    save_path = os.path.join(self.outdir, f'saved_model_epoch{epoch}')
                    results_log = os.path.join(self.outdir, 'results_log.csv')
                    torch.save(self.model.state_dict(), save_path)
                    log.info(f"Model saved to {sf.util.green(save_path)}")
                    metrics = sf.statistics.metrics_from_dataset(self.model,
                                                                model_type='categorical',
                                                                labels=self.labels,
                                                                patients=self.patients,
                                                                dataset=dataloaders['val'],
                                                                data_dir=self.outdir,
                                                                save_predictions=save_predictions)
                    # Log results
                    epoch_results = {'train_metrics': None, 'val_metrics': metrics }
                    for metric in metrics:
                        if metrics[metric]['tile'] is None: continue
                        epoch_results['tile'] = metrics[metric]['tile']
                        epoch_results['slide'] = metrics[metric]['slide']
                        epoch_results['patient'] = metrics[metric]['patient']
                    sf.util.update_results_log(results_log, 'trained_model', {f'epoch{epoch}': epoch_results})

        return {}

class LinearTrainer(Trainer):
    def __init__(self):
        raise NotImplementedError

class CPHTrainer(Trainer):
    def __init__(self):
        raise NotImplementedError