def get_character_names(args):
    if args.is_train:
        """
        Put the name of subdirectory in retargeting/datasets/Mixamo as [[names of group A], [names of group B]]
        """
        characters = [['Aj', 'BigVegas', 'Kaya', 'SportyGranny'],
                      ['Malcolm_m', 'Remy_m', 'Maria_m', 'Jasper_m', 'Knight_m',
                       'Liam_m', 'ParasiteLStarkie_m', 'Pearl_m', 'Michelle_m', 'LolaB_m',
                       'Pumpkinhulk_m', 'Ortiz_m', 'Paladin_m', 'James_m', 'Joe_m',
                       'Olivia_m', 'Yaku_m', 'Timmy_m', 'Racer_m', 'Abe_m']]

    else:
        """
        To run evaluation successfully, number of characters in both groups must be the same. Repeat is okay.
        """
        characters = [['BigVegas', 'BigVegas', 'BigVegas', 'BigVegas'],  ['Mousey_m', 'Goblin_m', 'Mremireh_m', 'Vampire_m']]
        tmp = characters[1][args.eval_seq]
        characters[1][args.eval_seq] = characters[1][0]
        characters[1][0] = tmp

    return characters

# For custom mixamo dataset
def get_character_names_custom(args):
    if args.is_train:
        """
        Put the name of subdirectory in retargeting/datasets/Mixamo as [[names of group A], [names of group B]]
        group A: model with source animation
        group B: model who is going to get motion retargetted
        """
        characters_A = open(f'./datasets/{args.dataset}/train_vox.txt').read().splitlines()
        characters_B = open(f'./datasets/{args.dataset}/val_vox.txt').read().splitlines()
        characters = [characters_A, characters_B]

    else:
        """
        To run evaluation successfully, number of characters in both groups must be the same. Repeat is okay.
        """
        characters_B = open(f'./datasets/{args.dataset}/vis_vox.txt').read().splitlines()
        # try just using group A ... from training set
        characters_A = open(f'./datasets/{args.dataset}/train_vox.txt').read().splitlines()
        # characters_A = [characters_B[0]]*len(characters_B)
        characters = [characters_A, characters_B]

        # switch eval_seq and the first element. In only the group B. why?
        tmp = characters[1][args.eval_seq]
        characters[1][args.eval_seq] = characters[1][0]
        characters[1][0] = tmp

    return characters

def create_dataset(args, character_names=None):
    from datasets.combined_motion import TestData, MixedData

    if args.is_train:
        return MixedData(args, character_names)
    else:
        return TestData(args, character_names)


# def get_test_set_bk():
#     with open('./datasets/Mixamo/test_list.txt', 'r') as file:
#         list = file.readlines()
#         list = [f[:-1] for f in list]
#         return list


# def get_test_set():
#     with open('./datasets/Mixamo_ori/test_list.txt', 'r') as file:
#         list = file.readlines()
#         list = [f[:-1] for f in list]
#         return list

def get_test_set(args):
    with open(f'./datasets/{args.dataset}/test_list.txt', 'r') as file:
        list = file.readlines()
        list = [f[:-1] for f in list]
        return list

# Not used anywhere...
# def get_train_list():
#     with open('./datasets/Mixamo/train_list.txt', 'r') as file:
#         list = file.readlines()
#         list = [f[:-1] for f in list]
#         return list
