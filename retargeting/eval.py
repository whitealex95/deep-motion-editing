import os
from models import create_model
from datasets import create_dataset, get_character_names, get_character_names_custom
import option_parser
import torch
from tqdm import tqdm


def eval(eval_seq, save_dir, test_device='cpu', epoch=200):
    para_path = os.path.join(save_dir, 'para.txt')
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]
        args = option_parser.get_parser().parse_args(argv_)

    args.cuda_device = test_device if torch.cuda.is_available() else 'cpu'
    args.is_train = False
    args.rotation = 'quaternion'
    args.eval_seq = eval_seq
    args.save_dir = save_dir
    # __import__('pdb').set_trace()
    if not args.use_original:
        character_names = get_character_names_custom(args)  # [['aj', 'aj', 'aj', 'aj'], ['aj', 'Ch14_nonPBR', 'kaya', 'mutant']]
    else:
        character_names = get_character_names(args)

    # TODO: enable target dataset to only have offsets
    dataset = create_dataset(args, character_names)

    model = create_model(args, character_names, dataset)
    model.load(epoch=epoch)
    # model.load(epoch=20000)
    # __import__('pdb').set_trace()
        
    for i, motions in tqdm(enumerate(dataset), total=len(dataset)):
        # no batching
        model.set_input(motions)
        model.test()


if __name__ == '__main__':
    parser = option_parser.get_parser()
    args = parser.parse_args()
    eval(args.eval_seq, args.save_dir, args.cuda_device)
