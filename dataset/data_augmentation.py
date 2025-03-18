import SimpleITK as sitk
import numpy as np
import scipy.ndimage
import random


def random_augmentation(arr_img, arr_mask,arr_mask1, shift_range, scale_range, gamma_range, p):
    if random.random() < p:
        arr_img = random_gamma_transformation(arr_img, gamma_range, p)
        arr_img, arr_mask,arr_mask1 = random_flip(arr_img, arr_mask,arr_mask1, p)
        # arr_img, arr_contrast_img, arr_mask = random_rotate(arr_img, arr_contrast_img, arr_mask, 3, p)
        arr_img, arr_mask,arr_mask1 = random_shift(arr_img, arr_mask,arr_mask1, shift_range, p)
        arr_img, arr_mask,arr_mask1 = random_scale(arr_img, arr_mask,arr_mask1, scale_range, p)
        arr_img  = random_noise(arr_img, p)
    return arr_img, arr_mask, arr_mask1


def random_rotate(arr_img,  arr_mask, angle_range, p):
    if random.random() < p:
        angle_x, angle_y, angle_z = random.uniform(-angle_range, angle_range), random.uniform(-angle_range,
                                                                                              angle_range), \
                                    random.uniform(-angle_range, angle_range)
        arr_img = rotate(arr_img, angle_x, "x", order=3)

        arr_mask = rotate(arr_mask, angle_x, "x", order=0)

        arr_img = rotate(arr_img, angle_y, "y", order=3)
        arr_mask = rotate(arr_mask, angle_y, "y", order=0)

        arr_img = rotate(arr_img, angle_z, "z", order=3)
        arr_mask = rotate(arr_mask, angle_z, "z", order=0)

    return arr_img,  arr_mask


def rotate(arr_img, angle, axial, order=3):
    if order == 0:
        cval = 0
    else:
        cval = np.percentile(arr_img, 1)
    if axial == "z":
        arr_img = scipy.ndimage.rotate(arr_img, angle, (1, 2), reshape=False, cval=cval, order=order)
    elif axial == "y":
        arr_img = scipy.ndimage.rotate(arr_img, angle, (0, 2), reshape=False, cval=cval, order=order)
    elif axial == "x":
        arr_img = scipy.ndimage.rotate(arr_img, angle, (0, 1), reshape=False, cval=cval, order=order)
    else:
        raise ValueError("axial must be one of x, y or z")
    return arr_img


def random_shift(arr_img,arr_mask,arr_mask1, shift_range, p):
    if random.random() < p:
        shift_z_range, shift_y_range, shift_x_range = shift_range
        shift_x, shift_y, shift_z = random.uniform(-shift_x_range, shift_x_range), \
                                    random.uniform(-shift_y_range, shift_y_range),\
                                    random.uniform(-shift_z_range, shift_z_range)
        shift_zyx = (shift_z, shift_y, shift_x)
        arr_img = shift(arr_img, shift_zyx, order=3)
        arr_mask = shift(arr_mask, shift_zyx, order=0)
        arr_mask1 = shift(arr_mask1, shift_zyx, order=0)
    return arr_img, arr_mask,arr_mask1


def shift(arr_img, shift, order):
    if order == 0:
        cval = 0
    else:
        cval = np.percentile(arr_img, 1)
    arr_img = scipy.ndimage.shift(arr_img, shift=shift, cval=cval, order=order)
    return arr_img


def random_flip(arr_img, arr_mask,arr_mask1, p):
    if random.random() < p:
        axial_list = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
        axial = random.choice(axial_list)
        for sub_axial in axial:
            arr_img = flip(arr_img, sub_axial)
            arr_mask = flip(arr_mask, sub_axial)
            arr_mask1 = flip(arr_mask1, sub_axial)
    return arr_img, arr_mask,arr_mask1


def flip(arr_img, axis):
    arr_img = np.flip(arr_img, axis=axis)
    return arr_img


# def random_permute(arr_img, arr_mask, p):
#     if random.random() < p:
#         axial_list = [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
#         permution = random.choice(axial_list)
#         arr_img = permute(arr_img, permution)
#         arr_mask = permute(arr_mask, permution)
#     return arr_img, arr_mask
#
#
# def permute(arr_img, permution):
#     arr_img = np.transpose(arr_img, axes=permution)
#     return arr_img


def random_noise(arr_image, p):
    if random.random() < p:
        arr_image = noise(arr_image)
    return arr_image


def noise(arr_image, ):
    std = np.std(arr_image)
    noise = np.random.random(arr_image.shape)
    noise = 0.1 * std * 2 * (noise - 0.5)
    arr_image = arr_image + noise
    return arr_image


def random_scale(arr_image, arr_mask,arr_mask1, scale_factor_range, p):
    if random.random() < p:
        low, hi = scale_factor_range
        scale_factor = random.uniform(low, hi)
        arr_image = scale(arr_image, scale_factor, order=3)
        arr_mask = scale(arr_mask, scale_factor, order=0)
        arr_mask1 = scale(arr_mask1, scale_factor, order=0)
    return arr_image, arr_mask,arr_mask1


def scale(arr_image, scale_factor, order):
    shapes = arr_image.shape
    if order == 0:
        cval = 0
    else:
        cval = np.percentile(arr_image, 1)
    arr_image = scipy.ndimage.zoom(arr_image, zoom=scale_factor, order=order, mode="constant", cval=cval)
    scaled_shapes = arr_image.shape
    for i, zip_data in enumerate(zip(shapes, scaled_shapes)):
        shape, scaled_shape = zip_data
        if scaled_shape < shape:
            padding = shape - scaled_shape
            left_padding = padding // 2
            right_padding = padding - left_padding
            if i == 0:
                arr_image = np.pad(arr_image, ((left_padding, right_padding), (0, 0),
                                               (0, 0)), constant_values=cval)
            elif i == 1:
                arr_image = np.pad(arr_image, ((0, 0), (left_padding, right_padding),
                                               (0, 0)), constant_values=cval)
            elif i == 2:
                arr_image = np.pad(arr_image, ((0, 0), (0, 0),
                                               (left_padding, right_padding)), constant_values=cval)
        elif scaled_shape > shape:
            crop = scaled_shape - shape
            left_crop = crop // 2
            right_crop = crop - left_crop
            if i == 0:
                arr_image = arr_image[left_crop: scaled_shape - right_crop, :, :]
            elif i == 1:
                arr_image = arr_image[:, left_crop: scaled_shape - right_crop, :]
            elif i == 2:
                arr_image = arr_image[:, :, left_crop: scaled_shape - right_crop]
    return arr_image


def random_gamma_transformation(arr_image, gamma_range, p):
    if random.random() < p:
        gamma = random.uniform(gamma_range[0], gamma_range[1])
        arr_image = gamma_transformation(arr_image, gamma)
    return arr_image


def gamma_transformation(arr_image, gamma):
    low, hi = arr_image.min(), arr_image.max()
    arr_image = (arr_image - low) / (hi - low)
    arr_image = np.power(arr_image, gamma)
    arr_image = (hi - low) * arr_image + low
    return arr_image


def load_nii_gz_as_array(nii_gz_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(nii_gz_path))


def write_array_as_nii_gz(arr_img, out_path):
    sitk.WriteImage(sitk.GetImageFromArray(arr_img), out_path)


if __name__ == '__main__':
    img_nii_gz_path = "../Cropped_Cyst_1.0_7_11/ct/C0002.nii.gz"
    mask_nii_gz_path = "../Cropped_Cyst_1.0_7_11/mask/C0002.nii.gz"
    arr_img = load_nii_gz_as_array(img_nii_gz_path)
    arr_mask = load_nii_gz_as_array(mask_nii_gz_path)
    # arr_img = scale(arr_img, 1.25, 3)
    # arr_mask = scale(arr_mask, 1.25, 0)
    # arr_img, arr_mask = random_permute(arr_img, arr_mask, 1)
    # arr_img, arr_mask = random_flip(arr_img, arr_mask, 1)
    # arr_img, arr_mask = random_shift(arr_img, arr_mask, 10, 1)
    arr_img, arr_mask = random_augmentation(arr_img, arr_mask,  (16, 16, 16), (0.75, 1.25), (0.7, 1.5), 1)
    write_array_as_nii_gz(arr_img, "test.nii.gz")
    write_array_as_nii_gz(arr_mask, "test_mask.nii.gz")
