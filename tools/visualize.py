import os
import torch
from torchvision.utils import save_image

def visualize(save_images_path, img, reconstr_img, masked_img, current_epoch, current_step):

    if not os.path.exists(save_images_path):
        os.makedirs(save_images_path, exist_ok=True)

    # generate images
    # with torch.no_grad():
        # fixed images

        # # decode again
        # cycreconst_rgb_images = base.generator_rgb.module.decode(fake_rgb_contents, real_rgb_styles)
        # cycreconst_ir_images = base.generator_ir.module.decode(fake_ir_contents, real_ir_styles)
        #
        # cycreconst2_rgb_images = base.generator_rgb.module.decode(real_rgb_contents, fake_rgb_styles)
        # cycreconst2_ir_images = base.generator_ir.module.decode(real_ir_contents, fake_ir_styles)
        #
        # wrong_fake_rgb_style, shuffled_fake_rgb_style = shuffle_styles(fake_rgb_styles, config.gan_k)
        # wrong_fake_ir_style, shuffled_fake_ir_style = shuffle_styles(fake_ir_styles, config.gan_k)
        #
        # cycreconst3_rgb_images = base.generator_rgb.module.decode(fake_rgb_contents, shuffled_fake_rgb_style)
        # cycreconst3_ir_images = base.generator_ir.module.decode(fake_ir_contents, shuffled_fake_ir_style)
        #
        # cycreconst4_rgb_images = base.generator_rgb.module.decode(fake_rgb_contents, wrong_fake_rgb_style)
        # cycreconst4_ir_images = base.generator_ir.module.decode(fake_ir_contents, wrong_fake_ir_style)

    # save images
    images = (torch.cat([img, reconstr_img, masked_img
                         # cycreconst_rgb_images, cycreconst_ir_images,
                         # cycreconst2_rgb_images, cycreconst2_ir_images,
                         # cycreconst3_rgb_images, cycreconst3_ir_images,
                         # cycreconst4_rgb_images, cycreconst4_ir_images
                         ], dim=0) + 1.0) / 2.0
    save_image(images.data.cpu(), os.path.join(save_images_path, '{}-{}.jpg'.format(current_epoch, current_step)), img.shape[0])