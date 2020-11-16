import os
from os.path import join as pjoin
from get_error import full_batch
import numpy as np
from option_parser import try_mkdir, get_parser
from eval import eval
import argparse


def batch_copy(source_path, suffix, dest_path, dest_suffix=None):
    try_mkdir(dest_path)
    files = [f for f in os.listdir(source_path) if f.endswith('_{}.bvh'.format(suffix))]

    length = len('_{}.bvh'.format(suffix))
    for f in files:
        if dest_suffix is not None:
            cmd = 'cp \"{}\" \"{}\"'.format(os.path.join(source_path, f), os.path.join(dest_path, f[:-length] + '_{}.bvh'.format(dest_suffix)))
        else:
            cmd = 'cp \"{}\" \"{}\"'.format(os.path.join(source_path, f), os.path.join(dest_path, f[:-length] + '.bvh'))
        os.system(cmd)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./pretrained/')
    parser.add_argument('--eval_epoch', type=int, default=30000)
    args = parser.parse_args()

    # Load trained args
    para_path = os.path.join(args.save_dir, 'para.txt')
    eval_epoch = args.eval_epoch
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]
        args = get_parser().parse_args(argv_)
    
    args.eval_epoch = eval_epoch
    prefix = args.save_dir
    print("args.use_original: ", args.use_original)
    if args.use_original:
        test_characters = ['Mousey_m', 'Goblin_m', 'Mremireh_m', 'Vampire_m']
    else:
        # test_characters = ['aj', 'Ch14_nonPBR', 'kaya', 'mutant']
        test_characters = open(f'./datasets/{args.dataset}/vis_vox.txt').read().splitlines()

    cross_dest_path = pjoin(prefix, 'results/cross_structure/')
    intra_dest_path = pjoin(prefix, 'results/intra_structure/')
    source_path = pjoin(prefix, 'results/bvh/')

    cross_error = []
    intra_error = []
    
    for i in range(4):  
        # what does 4 mean? it means we have 4 'from_{}' stuffs. 
        # however, it seems that the 'from_{}' are duplicated... is it? no!!! they are all different
        print('Batch [{}/4]'.format(i + 1))
        # evaluating with i-th character as source motion
        eval(i, prefix, epoch=args.eval_epoch)


        # cross error calculated only once
        print('Collecting test error...')
        if i == 0:
            cross_error += full_batch(args, 0, prefix)
            for char in test_characters:
                batch_copy(os.path.join(source_path, char), 0, os.path.join(cross_dest_path, char))
                batch_copy(os.path.join(source_path, char), 'gt', os.path.join(cross_dest_path, char), 'gt')

        # intra error calculated everytime  (shouldn't it be same?)
        intra_dest = os.path.join(intra_dest_path, 'from_{}'.format(test_characters[i]))
        for char in test_characters:
            for char in test_characters:
                batch_copy(os.path.join(source_path, char), 1, os.path.join(intra_dest, char))
                batch_copy(os.path.join(source_path, char), 'gt', os.path.join(intra_dest, char), 'gt')

        intra_error += full_batch(args, 1, prefix)

    cross_error = np.array(cross_error)
    intra_error = np.array(intra_error)

    cross_error_mean = cross_error.mean()
    intra_error_mean = intra_error.mean()

    os.system('rm -r %s' % pjoin(prefix, 'results/bvh'))

    print('Intra-retargeting error:', intra_error_mean)
    print('Cross-retargeting error:', cross_error_mean)
    print('Evaluation finished!')
