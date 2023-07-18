import argparse
import os
import lpips
import torchvision
from tqdm import tqdm
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir0)
all_scores = []

for file in tqdm(files, total=len(files)):
	if(os.path.exists(os.path.join(opt.dir1,file))):
		# Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))
		img0 = torchvision.transforms.functional.resize(img0, (256, 256), antialias=True)
		img1 = torchvision.transforms.functional.resize(img1, (256, 256), antialias=True)

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		all_scores.append(dist01.item())
		#print('%s: %.3f'%(file,dist01))
		f.writelines('%s: %.6f\n'%(file,dist01))

f.writelines(f'Average: {np.mean(all_scores)}')
f.close()
