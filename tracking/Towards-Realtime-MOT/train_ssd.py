import argparse
import json
import time

import test_ssd  
from models import *
from utils.datasets import JointDataset, collate_fn, ssd_JointDataset, ssd_collate_fn
from utils.utils import *
from utils.log import logger
from models_ssd import build_ssd_model

from ssd.config import cfg

def train(
        cfg,
        data_cfg,
        img_size=(1088,608),
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        freeze_backbone=True,
        opt=None,
):
    weights = '%s/weights' %cfg.OUTPUT_DIR
    mkdir_if_missing(weights) #k from utils.utils
    #latest = osp.join(weights, 'latest.pt')

    #239
    #latest = osp.join(weights, 'mobilenet_fpn_test_e30_b32_1e-3_5e-4_adam.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_e60_b32_1e-3_5e-4_adam.pt')
    #230
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e50_1e-3_Adam.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e60_1e-3_Adam.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e70_1e-3_Adam.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e70_1e-3_Adam_2.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e70_1e-3_Adam_8703.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_1024_b32_e70_1e-3_Adam.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e80_1e-3_Adam.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e80_1e-3_Adam_2.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e90_1e-3_Adam.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e90_1e-3_Adam_2.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e100_1e-3_Adam.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e100_1e-3_Adam_2.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e110_1e-3_Adam.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e110_1e-3_Adam_2.pt')

    #new
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e100_1e-3_Adam_new.pt')
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e100_1e-3_Adam_loss_test.pt')
    
    #1emb
    #latest = osp.join(weights, 'mobilenet_fpn_test_512_b32_e100_1e-3_Adam_1emb_gpu0.pt')
    latest = osp.join(weights, 'latest.pt')



    #m2det
    #latest = osp.join(weights, 'm2det_e30_b32_1e-3_5e-4_Adam.pt')


    torch.backends.cudnn.benchmark = True  # unsuitable for multiscale #k cudnn優化

    # Configure run
    f = open(data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()

    # Get dataloader
    dataset = ssd_JointDataset(cfg, dataset_root, trainset_paths, img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=8, pin_memory=True, drop_last=True, collate_fn=ssd_collate_fn) 
    
    # Initialize model
    model = build_ssd_model(cfg, dataset.nID)
    mkdir_if_missing(cfg.OUTPUT_DIR)

    start_epoch = 0
    if resume:
        checkpoint = torch.load(latest, map_location='cpu')

        # Load weights to resume from
        model.load_state_dict(checkpoint['model'])
        model.cuda().train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        del checkpoint  # current, saved

    else:
        model.cuda().train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        #optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)



    model = torch.nn.DataParallel(model)
    # Set scheduler
    #k 0.5*epoch降0.1 0.75*epoch再降0.1
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=[int(0.5*opt.epochs), int(0.75*opt.epochs)], gamma=cfg.SOLVER.GAMMA)

    model_info(model)
    t0 = time.time()
    #with torch.no_grad():
        #mAP, R, P = test_ssd.test(cfg, data_cfg, weights=latest, batch_size=cfg.TEST.BATCH_SIZE, img_size=img_size, print_interval=40, nID=dataset.nID)
        #test_ssd.test_emb(cfg, data_cfg, weights=latest, batch_size=cfg.TEST.BATCH_SIZE, img_size=img_size, print_interval=40, nID=dataset.nID)
    
    for epoch in range(epochs):
        
        epoch += start_epoch
        with open('./%s/log.txt' %cfg.OUTPUT_DIR, 'a') as output_log:
            output_log.write(('%8s%12s' + '%10s' * 6) % (
            'Epoch', 'Batch', 'box', 'conf', 'id', 'total', 'nTargets', 'time')+ '\n')
        with open('./%s/_check.txt' %cfg.OUTPUT_DIR, 'a') as output_log:
            output_log.write(('%8s%12s' + '%10s' * 6) % (
            'Epoch', 'Batch', 'box', 'conf', 'id', 'total', 'nTargets', 'time')+ '\n')
        logger.info(('%8s%12s' + '%10s' * 6) % (
            'Epoch', 'Batch', 'box', 'conf', 'id', 'total', 'nTargets', 'time'))


        # Freeze darknet53.conv.74 for first epoch
        if freeze_backbone and (epoch < 1):
            for name, p in model.named_parameters():
                if 'module.backbone.features' in name:
                    p.requires_grad = False if (epoch == 0) else True

        ui = -1
        rloss = defaultdict(float)  # running loss
        optimizer.zero_grad()
        for i, (imgs, targets, img_path, _, targets_len) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue

            # SGD burn-in
            burnin = min(1000, len(dataloader))
            if (epoch == 0) & (i <= burnin):
                lr = opt.lr * (i / burnin) **4 
                for g in optimizer.param_groups:
                    g['lr'] = lr

            targets['img_path'] = img_path
            targets['labels'].cuda()
            targets['ids'].cuda()
            targets['boxes'].cuda()

            # Compute loss, compute gradient, update parameters
            loss, components = model(imgs.cuda(), targets, targets_len.cuda())
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1
            
            for ii, key in enumerate(model.module.box_head.loss_names):
                rloss[key] = (rloss[key] * ui + components[ii]) / (ui + 1)

            s = ('%8s%12s' + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, len(dataloader) - 1),
                rloss['box'], rloss['conf'],
                rloss['id'],rloss['loss'],
                rloss['nT'], time.time() - t0)
            t0 = time.time()
            if i % opt.print_interval == 0:
                with open('./%s/log.txt' %cfg.OUTPUT_DIR, 'a') as output_log:
                    output_log.write(s+'\n')
                with open('./%s/_check.txt' %cfg.OUTPUT_DIR, 'a') as output_log:
                    output_log.write(s+'\n')
                logger.info(s)

        # Update scheduler (automatic)
        scheduler.step()
        with open('./%s/_img_path.txt' %cfg.OUTPUT_DIR, 'a') as output_log:
             output_log.write('='*30+'\n')

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'model': model.module.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, latest)


        # Calculate mAP
        if (epoch+1) % opt.test_interval ==0:
            checkpoint = {'epoch': epoch,
                          'model': model.module.state_dict(),
                          'optimizer': optimizer.state_dict()}
            epoch_weight = osp.join(weights, 'epoch_%d.pt'%epoch)
            torch.save(checkpoint, epoch_weight)
            with torch.no_grad():
                mAP, R, P = test_ssd.test(cfg, data_cfg, weights=latest, batch_size=batch_size, img_size=img_size, print_interval=80, nID=dataset.nID)
               #test_ssd.test_emb(cfg, data_cfg, weights=latest, batch_size=batch_size, img_size=img_size, print_interval=40, nID=dataset.nID)
    checkpoint = {'model': model.module.state_dict()}
    final = osp.join(weights, 'final.pt')
    torch.save(checkpoint, final)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe.json', help='coco.data file path')
    parser.add_argument('--img-size', type=int, default=(512, 512), help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--print-interval', type=int, default=40, help='print interval')
    parser.add_argument('--test-interval', type=int, default=10, help='test interval')
    parser.add_argument('--lr', type=float, default=1e-2, help='init lr')
    parser.add_argument('--unfreeze-bn', action='store_true', help='unfreeze bn')
    opt = parser.parse_args()

    init_seeds()

    #ssd_cfg_path = './vgg_ssd512_voc0712.yaml'
    ssd_cfg_path = './mobilenet_fpn_test_512_bdd100k_E200.yaml'
    #ssd_cfg_path = './mobilenet_fpn7_test.yaml'
    #ssd_cfg_path = './vgg_m2det_512.yaml'
    #ssd_cfg_path = './mobilenet_fpn_test_1024.yaml'
    #ssd_cfg_path = './mobilenet_fpn7.yaml'

    cfg.merge_from_file(ssd_cfg_path)
    cfg.freeze()

    train(
        cfg,
        opt.data_cfg,
        img_size=(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE),
        resume=opt.resume,
        epochs=cfg.SOLVER.EPOCH,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        accumulated_batches=opt.accumulated_batches,
        opt=opt,
    )
