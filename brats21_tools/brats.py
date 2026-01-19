import os
import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset
import glob


class BraTS21Dataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.samples = []
        self.mode = mode
        self.target_size = (128, 128, 128)
        self.epsilon = 1e-8


        seg_files = []
        for fold in os.listdir(data_dir):
            fold_path = os.path.join(data_dir, fold)
            if not os.path.isdir(fold_path):
                continue

            seg_files.extend(glob.glob(os.path.join(fold_path, '**', '*_seg.nii.gz'), recursive=True))


        for seg_path in seg_files:
            case_id = os.path.basename(seg_path).split('_seg.nii.gz')[0]
            case_dir = os.path.dirname(seg_path)

            npy_image_path = os.path.join(case_dir, f"{case_id}_image.npy")
            npy_mask_path = os.path.join(case_dir, f"{case_id}_mask.npy")
            paths = {
                # 'flair': os.path.join(case_dir, f"{case_id}_flair.nii.gz"),
                # 't1': os.path.join(case_dir, f"{case_id}_t1.nii.gz"),
                # 't1ce': os.path.join(case_dir, f"{case_id}_t1ce.nii.gz"),
                # 't2': os.path.join(case_dir, f"{case_id}_t2.nii.gz"),
                # 'seg': seg_path,
                'npy_image': npy_image_path,
                'npy_mask': npy_mask_path
            }

            self.samples.append(paths)

        self.transform = self.build_transform()


    def __len__(self):
        return len(self.samples)

    def build_transform(self):
        if self.mode == 'test':
            return tio.Compose([
                tio.ZNormalization(masking_method=tio.ZNormalization.mean, exclude=['mask']),
                tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=(0.5, 99.5), exclude=['mask']),
            ])
        else:

            return tio.Compose([
                tio.RandomAffine(
                    scales=(0.9, 1.1),
                    degrees=5,
                    translation=(3, 3, 3),
                    p=0.5
                ),
                tio.RandomFlip(axes=(0, 1, 2), p=0.5),
                tio.RandomMotion(p=0.2),
                tio.RandomBiasField(p=0.3),
                tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
                tio.RandomBlur(p=0.2),
                tio.RandomNoise(p=0.2),
                tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                tio.RescaleIntensity(out_min_max=(-1, 1), percentiles=(0.5, 99.5), exclude=['mask'])])

    def safe_save_npy(self, data, save_path, max_attempts=3):

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir, exist_ok=True)

            except PermissionError:
                raise Exception(f"no : {save_dir}")
            except Exception as e:
                raise Exception(f"failed: {str(e)}")


        for attempt in range(max_attempts):
            try:

                if not isinstance(data, np.ndarray):
                    data = np.asarray(data)
                if data.dtype == np.float16:
                    data = data.astype(np.float32)


                np.save(save_path, data)

                if os.path.exists(save_path):
                    print('ok!')
                else:
                    print('failed!')

                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path)
                    if file_size < 1024:
                        raise Exception(f"file is small: （{file_size} bytes）")
                    print(f"saved npy（size: {file_size / 1024 ** 2:.2f} MB）: {save_path}")
                    return True
                else:
                    raise Exception("file is not found")

            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"trying saving {attempt + 1} failed: {str(e)}，again...")
                    continue
                else:
                    raise Exception(f"via {max_attempts} times for saving failed: {str(e)}")

        return False

    def process_and_save_npy(self, paths):

        try:

            flair = tio.ScalarImage(paths['flair']).data  # (1, H, W, D)
            t1 = tio.ScalarImage(paths['t1']).data
            t1ce = tio.ScalarImage(paths['t1ce']).data
            t2 = tio.ScalarImage(paths['t2']).data
            seg = tio.LabelMap(paths['seg']).data  # (1, H, W, D)
        except Exception as e:
            raise Exception(f"onload original data failed: {str(e)}")


        image = torch.cat([flair, t1, t1ce, t2], dim=0)
        seg_data = seg[0].long()


        non_zero = torch.nonzero(seg_data, as_tuple=False)
        if len(non_zero) == 0:

            h, w, d = seg_data.shape
            center = (h // 2, w // 2, d // 2)
        else:
            min_coords = non_zero.min(dim=0).values
            max_coords = non_zero.max(dim=0).values
            center = ((min_coords + max_coords) // 2).tolist()


        start = [
            max(0, center[0] - self.target_size[0] // 2),
            max(0, center[1] - self.target_size[1] // 2),
            max(0, center[2] - self.target_size[2] // 2)
        ]
        end = [
            min(start[0] + self.target_size[0], seg_data.shape[0]),
            min(start[1] + self.target_size[1], seg_data.shape[1]),
            min(start[2] + self.target_size[2], seg_data.shape[2])
        ]


        for i in range(3):
            if end[i] - start[i] < self.target_size[i]:
                start[i] = max(0, end[i] - self.target_size[i])


        image_cropped = image[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        seg_cropped = seg_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]


        pad = []
        for i in range(3):
            current_size = image_cropped.shape[i + 1]
            if current_size < self.target_size[i]:
                pad_pre = (self.target_size[i] - current_size) // 2
                pad_post = self.target_size[i] - current_size - pad_pre
                pad.extend([pad_pre, pad_post])
            else:
                pad.extend([0, 0])

        if any(pad):

            image_cropped = torch.nn.functional.pad(
                image_cropped, pad[::-1], mode='replicate'
            )
            seg_cropped = torch.nn.functional.pad(
                seg_cropped.unsqueeze(0), pad[::-1], mode='constant', value=0
            ).squeeze(0)

        # （1→1, 2→2, 4→3）
        new_mask = torch.zeros_like(seg_cropped)
        new_mask[seg_cropped == 1] = 1
        new_mask[seg_cropped == 2] = 2
        new_mask[seg_cropped == 4] = 3


        image_np = image_cropped.cpu().numpy().copy()
        mask_np = new_mask.cpu().numpy().copy()


        self.safe_save_npy(image_np, paths['npy_image'])
        self.safe_save_npy(mask_np, paths['npy_mask'])

        return image_cropped, new_mask

    def __getitem__(self, idx):
        paths = self.samples[idx]


        if os.path.exists(paths['npy_image']) and os.path.exists(paths['npy_mask']):
            try:

                image = torch.from_numpy(np.load(paths['npy_image'])).float()
                mask = torch.from_numpy(np.load(paths['npy_mask'])).long()

            except Exception as e:

                image, mask = self.process_and_save_npy(paths)
        else:
            try:
                image, mask = self.process_and_save_npy(paths)
            except Exception as e:
                return self.__getitem__((idx + 1) % len(self))

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image),
            mask=tio.LabelMap(tensor=mask.unsqueeze(0))
        )
        transformed = self.transform(subject)
        image = transformed.image.data.float()
        mask = transformed.mask.data.long()[0]

        if mask.sum() == 0:
            return self.__getitem__((idx + 1) % len(self))

        if torch.isnan(image).any() or torch.isinf(image).any():
            return self.__getitem__((idx + 1) % len(self))

        return image, mask



if __name__ == "__main__":
    data_dir = r"F:\experiment\dataset\brats2021\test"
    dataset = BraTS21Dataset(data_dir, mode='train')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for images, masks in loader:
        print(f"data shape: {images.shape}, label shape: {masks.shape}")
        # break
