import os
import shutil
from tqdm import tqdm

import numpy as np
import torch

from core.data_loader import get_inf_loader
from core import utils


@torch.no_grad()
def inference_from_latent(nets, args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    domains = os.listdir(args.val_img_dir)
    domains.sort()
    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)

    for trg_idx, trg_domain in enumerate(domains):

        src_domains = [x for x in domains if x != trg_domain]

        for src_idx, src_domain in enumerate(src_domains):

            path_src = os.path.join(args.val_img_dir, src_domain)

            if len(os.listdir(path_src)) == 0:
                continue
            
            loader_src = get_inf_loader(root=path_src,
                                        img_size=args.img_size,
                                        batch_size=args.val_batch_size,
                                        imagenet_normalize=False)

            task = '%s2%s' % (src_domain, trg_domain)
            path_fake = os.path.join(args.eval_dir, task)
            shutil.rmtree(path_fake, ignore_errors=True)
            os.makedirs(path_fake)

            print('Generating images for %s...' % task)

            for i, batch in enumerate(tqdm(loader_src, total=len(loader_src))):

                x_src, fnames = batch
                print(fnames)
                N = x_src.size(0)
                x_src = x_src.to(device)
                y_trg = torch.tensor([trg_idx] * N).to(device)
                masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

                # generate 10 outputs from the same input
                group_of_images = []

                for j in range(args.num_outs_per_domain):
                    
                    z_trg = torch.randn(N, args.latent_dim).to(device)
                    s_trg = nets.mapping_network(z_trg, y_trg)

                    x_fake = nets.generator(x_src, s_trg, masks=masks)
                    group_of_images.append(x_fake)

                    # save generated images
                    for k in range(N):
                        filename = os.path.join(
                            path_fake,
                            f'{i*args.val_batch_size+(k+1)}_{j+1}_{fnames[k]}.png')
                        utils.save_image(x_fake[k], ncol=1, filename=filename)
