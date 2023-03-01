import os
from matplotlib import pyplot as plt
import torchvision
import numpy as np


def get_class_attr(Cls) -> []:
    """
    get attribute name from Class(type)
    :param Cls:
    :return:
    """
    import re
    return [a for a, v in Cls.__dict__.items()
            if not re.match('<function.*?>', str(v))
            and not (a.startswith('__') and a.endswith('__'))]


def get_class_attr_val(cls):
    """
    get attribute name and their value from class(variable)
    :param cls:
    :return:
    """
    attr = get_class_attr(type(cls))
    attr_dict = {}
    for a in attr:
        attr_dict[a]


def get_output_folder(parent_dir, env_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's saving directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def show_img(origin_image, attack_image):

    UNmean = [-1, -1, -1]
    UNstd = [2, 2, 2]

    UnNormalize = torchvision.transforms.Normalize(mean=UNmean, std=UNstd)
    origin_image = UnNormalize(origin_image)
    attack_image = UnNormalize(attack_image)
    img = torchvision.utils.make_grid(origin_image).numpy()
    plt.subplot(1, 2, 1)  # 表示第i张图片，下标只能从1开始，不能从0，
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.title("Origin")

    plt.subplot(1, 2, 2)  # 表示第i张图片，下标只能从1开始，不能从0，
    img = torchvision.utils.make_grid(attack_image).numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.title("Attack")

    plt.show()  # 展示原图片
