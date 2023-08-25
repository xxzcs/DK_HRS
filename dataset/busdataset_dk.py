import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .randaugment import RandAugmentMC

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

mean=(0.371, 0.371, 0.372)
std=(0.167, 0.167, 0.167)

# python train_bus.py --gpu 0 --out bus@878_b8_u20 --lambda-u 20 --lr 0.0046875 | tee b8_un30_lambdauw20_sgd_369.txt

def get_bus(args, root):
    
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    train_labeled_dataset = Bcancer(
        traindir, args,
        transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        db_path=args.labeledpath,
        )

    train_unlabeled_dataset = Bcancer(
        traindir, args,
        transform=TransformFixMatch(mean,std),
        db_path=args.unlabeledpath,
        is_unlabeled=True,
        )

    val_dataset = datasets.ImageFolder(
        valdir, 
        transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)]))
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip()])
        self.strong = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def load_db(db_path, class_to_idx):
    db = torch.load(db_path)

    images = []

    for key in sorted(db.keys()):

        for image_path in db[key]:
            image_path = '../' + image_path
            images.append((image_path, class_to_idx[key]))
    return images

def generate_aug_img(img, img_path,i):

    dk_img_dir_list = os.listdir('../dkimg_mean')
    # aug_func = ImageNetPolicy()
    # i = random.randint(0, len(dk_img_dir_list))
    # dk_img_name = ''
    sub_dir = dk_img_dir_list[i]
    dk_img_dir = os.path.join('../dkimg_mean', dk_img_dir_list[i])
    if sub_dir == 'manual_roi_img_mean':
        dk_img_name = os.path.basename(img_path)[0:-4] + '_roi.jpg'
    elif sub_dir == 'manual_roi15_img_mean':
        dk_img_name = os.path.basename(img_path)[0:-4] + '_roi15.jpg'
    else:
        dk_img_dir = '../dkimg_mean15'
        dk_img_name = os.path.basename(img_path)[0:-4] + '_margin_reinforce_roi15.jpg'
            
    dk_img_name = os.path.join(dk_img_dir, dk_img_name)
    dk_img = default_loader(dk_img_name)

    return dk_img

class Bcancer(data.Dataset):
    def __init__(self, root, args, transform=None, target_transform=None,
                 loader=default_loader, db_path='./data_split/labeled_images_1541.00.pth', is_unlabeled=False):
        classes, class_to_idx = find_classes(root)
        #imgs = make_dataset(root, class_to_idx)
        
        imgs = load_db(db_path, class_to_idx)

        random.shuffle(imgs)


        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.is_unlabeled = is_unlabeled
        self.autoaugment = generate_aug_img

        self.indices = [i for i in range(len(imgs))]
        random.shuffle(self.indices)
        # whether to preserve in the new version
        '''
        if self.is_unlabeled:
            self.total_train_count = args.batch_size_unlabeled * args.max_iter * args.unlabeled_iter
        else:
            self.total_train_count = args.batch_size * args.max_iter

        print("sample count {}".format(len(self.indices)))
        print("total sample count {}".format(self.total_train_count))
        '''
        self.total_train_count = len(imgs)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        #if self.is_unlabeled:
        #    print ("reading index {}".format(index))
        random_index = self.indices[index % len(self.indices)]
        path, target = self.imgs[random_index]
        img = self.loader(path)
        # print("###################before############")
        # print(img.size)
        # if self.is_unlabeled:
        #     weak, strong = self.autoaugment(img)
        
        if self.is_unlabeled:
            weak, strong = self.transform(img)
            # generate dk augmented image for each image 
            width = 3
            alpha = 1.
            mix_number = 3
            ws = np.float32(np.random.dirichlet([alpha]*width))
            # aug_ws = np.float32(np.random.dirichlet([alpha]*width))
            m = np.float32(np.random.beta(alpha,alpha))
            aug_img = torch.zeros_like(strong)
            aug_img_dk = Image.new("RGB", (img.width,img.height))
            for i in range(width):
                dk_aug_img = self.autoaugment(img, path,i)
                aug_img_dk += ws[i] * dk_aug_img
            aug_img_dk = Image.fromarray(np.uint8(aug_img_dk)) 

            new_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
            # aug_img =  (1 - m) * strong + m * new_transform(aug_img_dk)
            # aug_img =  0.55 * strong + 0.45 * 
            # newweak = strong
            # newstrong = aug_img
            # dk_strong = aug_img
            dkimg = new_transform(aug_img_dk)
        else:
            img = self.transform(img)
        
        # if self.transform is not None:
        #     img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.is_unlabeled:
            return weak, strong, dkimg
            # return weak, strong
        else:
            return img, target
        
        # print("###################after############")
        # print(img.shape)
        # return img, target

    def __len__(self):
        return self.total_train_count

