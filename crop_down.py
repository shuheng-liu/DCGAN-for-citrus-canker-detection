import PIL
import os, sys
from PIL import Image

def handle_folder(dir_raw, dir_new, grid_size = 64, margin_size = 2, max_imgs = -1):
    assert os.path.isdir(dir_raw)
    legal_suffices = ['jpg', 'JPG', 'png', 'PNG']
    try: os.mkdir(dir_new)
    except: pass
    img_names = os.listdir(dir_raw)
    img_names.sort()
    img_count = 0
    for img_name in img_names:
        if img_name.split('.')[-1] not in legal_suffices: continue
        img_count += 1
        if 0 <= max_imgs < img_count: break
        print('processing', img_name)
        img = Image.open(os.path.join(dir_raw, img_name))
        width, height = img.size
        assert (width - margin_size) % (margin_size + grid_size) == 0, (width, margin_size, grid_size)
        assert (height - margin_size) % (margin_size + grid_size) == 0, (height, margin_size, grid_size)
        num_w = (width - margin_size) // (grid_size + margin_size)
        num_h = (height - margin_size) // (grid_size + margin_size)
        for i in range(num_w):
            for j in range(num_h):
                x0, y0 = margin_size + i * (grid_size + margin_size),  margin_size + j * (grid_size + margin_size)
                x1, y1 = x0 + grid_size, y0 + grid_size
                grid = img.crop([x0, y0, x1, y1])
                save_name = os.path.splitext(img_name)[0] + "_%d_%d"%(i,j) + os.path.splitext(img_name)[1]
                grid.save(os.path.join(dir_new, save_name))

def make_list(dir, txt_dir = '/Users/liushuheng/Desktop'):
    assert os.path.isdir(dir)
    assert os.path.isdir(txt_dir)
    legal_suffices = ['jpg', 'JPG', 'png', 'PNG']
    img_names = os.listdir(dir)
    img_names.sort()
    with open(os.path.join(txt_dir, 'list.txt'), 'w') as f:
        for img_name in img_names:
            if img_name.split('.')[-1] not in legal_suffices: continue
            if 'generate' in img_name:
                tag = 0
            else:
                tag = 1
            f.write('{} {}\n'.format(img_name, tag))

def resize_real(dir_raw, dir_new, grid_size = 64):
    assert os.path.isdir(dir_raw)
    legal_suffices = ['jpg', 'JPG', 'png', 'PNG']
    try: os.mkdir(dir_new)
    except: pass
    img_names = os.listdir(dir_raw)
    img_names.sort()
    for img_name in img_names:
        print('processing', img_name)
        if img_name.split('.')[-1] not in legal_suffices: continue
        img = Image.open(os.path.join(dir_raw, img_name))
        img_resized = img.resize((grid_size, grid_size), Image.ANTIALIAS)
        img_resized.save(os.path.join(dir_new, img_name))


if __name__ == "__main__":
    paths = ['/Users/liushuheng/Desktop/dcgan-187',
             '/Users/liushuheng/Desktop/dcgan-188',
             '/Users/liushuheng/Desktop/dcgan-193',
             '/Users/liushuheng/Desktop/dcgan-194']
    real_path = '/Users/liushuheng/Documents/Computer Vision 2/TestPicture/PositiveSamplesHQ'
    for path in paths:
        grid_path = path + '-grid'
        handle_folder(path, grid_path, max_imgs=10)
        resize_real(real_path, grid_path)
        make_list(grid_path)
