import sys
import os
import types
import time
import torch
import yaml
import random
import logging
import slideflow as sf
import numpy as np
import slideflow.slide
import slideflow.statistics

from os.path import join
from PIL import Image
from slideflow.model import base as _base
from slideflow.model.adv_xception import xception_fc
from slideflow.util import log, StainNormalizer
from slideflow.model import torch_utils
from slideflow.model.utils import log_manifest
from tqdm import tqdm
from slideflow.clam.models.model_clam import Attn_Net_Gated

sys.path.insert(0, '/home/shawarma/yolov5')
from models.yolo import Model as YoloV5
from utils.augmentations import letterbox

SLIDE_PATH = '/home/shawarma/256632.svs'

class AttentionNet(torch.nn.Module):
    def __init__(self, n_classes):
        super(AttentionNet, self).__init__()
        self.attention_net = Attn_Net_Gated(L=2048, D=256, dropout=True, n_classes=1)
        self.classifier = torch.nn.Linear(2048, n_classes)

    def forward(self, h):
        A, h = self.attention_net(h) # NxK
        A = torch.transpose(A, 1, 0) # KxN
        A_softmax = torch.nn.functional.softmax(A, dim=1)
        M = torch.mm(A_softmax, h)
        logits = self.classifier(M)
        #Y_hat = torch.topk(logits, 1, dim=1)[1]
        #Y_prob = torch.nn.functional.softmax(logits, dim=1)
        return logits, A#, Y_prob, Y_hat, A

class SlideDetector:
    def __init__(self, slide_path, yolo, device, img_size=299):
        self.img_size=img_size
        self.wsi = sf.slide.WSI(slide_path, 299, 302)
        self.thumb = self.wsi.thumb()
        self.thumb_x_scale = self.wsi.slide.dimensions[0] / self.thumb.size[0]
        self.thumb_y_scale = self.wsi.slide.dimensions[1] / self.thumb.size[1]
        self.yolo = yolo

    def read_img(self, sx, sy, ex, ey, scale='full'):
        '''x and y in full resolution coordinates'''
        assert scale in ('full', 'thumb')

        if scale == 'thumb':
            tx, ty = self.thumb_x_scale, self.thumb_y_scale
            sx, sy, ex, ey = sx * tx, sy * ty, ex * tx, ey * ty

        width_x = ex-sx
        width_y = ey-sy
        greatest = max(width_x, width_y)
        downsample = greatest / self.img_size
        level = self.wsi.slide.get_best_level_for_downsample(downsample)
        downsample_factor = self.wsi.slide.level_downsamples[level]
        width_x /= downsample_factor
        width_y /= downsample_factor

        region = self.wsi.slide.read_region((sx, sy), level, [width_x, width_y]) # Downsample coordinates
        region = region.thumbnail_image(self.img_size) # target resolution coordinates
        if region.bands == 4:
            region = region.flatten() #(removes alpha)
        image = sf.slide.vips2numpy(region)
        image, _, _ = letterbox(image, new_shape=(self.img_size, self.img_size), stride=self.img_size)
        image = image.transpose(2, 0, 1) #HWC -> CWH
        image = image / 127.5 - 1
        return image

    def get_batch(self, n):
        # Fake predictions, just for testing
        batch = []
        '''results = self.yolo(self.thumb) #results.pred = (xy, xy, softmax, class) for each object

        batch = []
        for result in results.pred:
            if result.shape[0]:
                x, y, ex, ey, s, c = result
                s_x = self.thumb_x_scale
                s_y = self.thumb_y_scale
                image = self.read_img(x*s_x, y*s_y, ex*s_x, ey*s_y)
                batch.append(image)'''
        results = []
        for i in range(n - len(results)):
            xw = max(min((np.random.lognormal(0, 1)/100), 1), 0)
            yw = max(min((np.random.lognormal(0, 1)/100), 1), 0)
            x_width = int(max(xw * (self.wsi.slide.dimensions[0]-2), self.img_size))
            y_width = int(max(yw * (self.wsi.slide.dimensions[0]-2), self.img_size))

            # Limit width of longest dimension to 2 times the size of the smallest dimension
            x_width = min(2*y_width, x_width)
            y_width = min(2*x_width, y_width)

            rx = random.randint(0, self.wsi.slide.dimensions[0]-(x_width+1))
            ry = random.randint(0, self.wsi.slide.dimensions[1]-(y_width+1))
            image = self.read_img(rx, ry, rx+x_width, ry+y_width)
            batch.append(image)
        import time
        time.sleep(10000)
        return np.array(batch)

class AttentionTrainer:
    def __init__(self, hp, outdir, labels, patients, name=None, manifest=None, slide_input=None, feature_sizes=None,
                 feature_names=None, outcome_names=None, normalizer=None, normalizer_source=None, mixed_precision=True):

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

    def train(self, train_dts, val_dts, sites_dict, num_sites, site_lambda=10, validate_on_batch=512, validation_batch_size=32, validation_steps=100,
              save_predictions=True, skip_metrics=False, seed=0, log_frequency=None, starting_epoch=0,
              ema_observations=None, ema_smoothing=None, use_tensorboard=None, steps_per_epoch_override=0,
              multi_gpu=None, resume_training=None, pretrain='imagenet', checkpoint=None):

        # Training preparation
        log_manifest(train_dts.tfrecords(), val_dts.tfrecords(), self.labels, join(self.outdir, 'slide_manifest.csv'))
        device = torch.device("cuda")
        starting_epoch = max(starting_epoch, 1)
        if steps_per_epoch_override:
            steps_per_epoch = steps_per_epoch_override
            log.info(f"Overriding steps per epoch = {steps_per_epoch_override}")
        else:
            steps_per_epoch = train_dts.num_tiles // self.hp.batch_size
            log.info(f"Steps per epoch = {steps_per_epoch}")

        # Build models
        num_classes = self.hp._detect_classes_from_labels(self.labels)[0]
        feature_G = xception_fc().to(device)
        att_D = AttentionNet(n_classes=num_classes).to(device)

        # Outcome optimizer
        params = list(feature_G.parameters()) + list(att_D.parameters())
        optimizer = self.hp.get_opt(params)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Slide detector
        #yolo_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s',)
        with open('/home/shawarma/yolov5/data/hyps/hyp.scratch.yaml', errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        yolo_detector = YoloV5('/home/shawarma/yolov5/models/yolov5s_sf.yaml', ch=3, nc=1, anchors=hyp.get('anchors')).autoshape().to(device)
        logging.getLogger('pyvips').setLevel(50)
        slide_detector = SlideDetector(SLIDE_PATH, yolo_detector, device)

        scaler = torch.cuda.amp.GradScaler()

        # Dataloaders
        interleave_args = types.SimpleNamespace(
            rank=0,
            num_replicas=1,
            labels=self.labels,
            seed=seed,
            chunk_size=16,
            normalizer=self.normalizer,
            pin_memory=True,
            num_workers=4,
            onehot=False,
            rebuild_index=False,
        )
        dataloaders = {
            'train': iter(train_dts.torch(infinite=True, batch_size=self.hp.batch_size, augment=True, incl_slidenames=True, **vars(interleave_args))),
            'val': val_dts.torch(infinite=False, batch_size=validation_batch_size, augment=False, incl_slidenames=True, **vars(interleave_args))
        }

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
                    feature_G.train()
                    att_D.train()

                    num_steps = steps_per_epoch * self.hp.batch_size
                    dataloader_pb = tqdm(total=num_steps, ncols=100, unit='img', leave=False)

                    while step < steps_per_epoch:
                        #images, labels, slides = next(dataloaders['train'])
                        #images = images.to(device, non_blocking=True)
                        #labels = torch.unsqueeze(labels[0], 0).to(device, non_blocking=True)
                        images = torch.from_numpy(slide_detector.get_batch(self.hp.batch_size)).half().to(device)
                        labels = torch.tensor([0]).to(device)

                        optimizer.zero_grad()

                        with torch.cuda.amp.autocast():
                            features = feature_G(images)
                            outputs, A = att_D(features)
                            loss = loss_fn(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        # Update running totals and losses
                        running_loss += loss.item() * images.size(0)
                        num_correct = torch.sum(preds == labels.data)
                        running_corrects += num_correct

                        num_records = step * self.hp.batch_size
                        dataloader_pb.set_description(f'{phase} acc: {running_corrects / num_records:.4f}')
                        dataloader_pb.update(self.hp.batch_size)
                        step += 1
                    dataloader_pb.close()

                # Perform basic evaluation at every epoch end
                if phase == 'val' and (val_dts is not None):
                    feature_G.eval()
                    att_D.eval()

                    dataloader_pb = tqdm(total=dataloaders['val'].num_tiles, ncols=100, unit='img', leave=False)
                    for images, labels, slides in dataloaders['val']:
                        images = images.to(device, non_blocking=True)
                        labels = torch.unsqueeze(labels[0], 0).to(device, non_blocking=True)

                        optimizer.zero_grad()
                        with torch.no_grad():
                            with torch.cuda.amp.autocast() if self.mixed_precision else sf.model.utils.no_scope():
                                features = feature_G(images)
                                outputs, A = att_D(features)
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
                    feature_G.eval()
                    att_D.eval()
                    model = torch.nn.Sequential(feature_G, att_D)

                    save_path = os.path.join(self.outdir, f'saved_model_epoch{epoch}')
                    torch.save(model.state_dict(), save_path)
                    log.info(f"Model saved to {sf.util.green(save_path)}")
                    metrics = sf.statistics.metrics_from_dataset(model,
                                                                model_type='categorical',
                                                                labels=self.labels,
                                                                patients=self.patients,
                                                                dataset=dataloaders['val'],
                                                                data_dir=self.outdir,
                                                                save_predictions=save_predictions)