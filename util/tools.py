
import SimpleITK as sitk
import numpy as np
def save_numpy_as_nii(data:None,targets=None, vols=None):
    shape = (240, 240, 155)
    new_spacing = (shape[0]/160, shape[1]/160, shape[2]/160)

    new_shape = (160, 160, 160)
    direction =(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    origin=(0.0, -239.0, 0.0)

    data = np.asarray(data, dtype='uint8')
    for i in range(0, 4):

            tmp = np.squeeze(data, axis=0)[i,:,:,:]
            # tmp[tmp>=10]=0

            image = sitk.GetImageFromArray(tmp)
            image.SetDirection(direction)
            image.SetOrigin(origin)
            image.SetSpacing(new_spacing)
            sitk.WriteImage(image, './data'+str(i)+'.nii')

    seg_image = sitk.GetImageFromArray(targets[0])
    seg_image.SetDirection(direction)
    seg_image.SetOrigin(origin)
    seg_image.SetSpacing(new_spacing)
    sitk.WriteImage(seg_image, './data_seg.nii')

    for index, modal in enumerate(vols[0]):
        vol_image = sitk.GetImageFromArray(modal)
        vol_image.SetDirection(direction)
        vol_image.SetOrigin(origin)
        vol_image.SetSpacing(new_spacing)
        sitk.WriteImage(vol_image, './data_vol_' + str(index) + '.nii')



