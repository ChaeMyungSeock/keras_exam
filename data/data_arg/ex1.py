import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage


images = np.random.randint(0, 50, (4, 128, 128, 3), dtype=np.uint8)

# Generate random keypoints, 1-10 per image with float32 coordinates
keypoints = []
for image in images:
    n_keypoints = np.random.randint(1, 10)
    kps = np.random.random((n_keypoints, 2))
    kps[:, 0] *= image.shape[0]
    kps[:, 1] *= image.shape[1]
    keypoints.append(kps)

seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)),
                      iaa.Affine(scale=(0.5, 0.7))])

# augment keypoints and images
images_aug, keypoints_aug = seq(images=images, keypoints=keypoints)

# Example code to show each image and print the new keypoints coordinates
for i in range(len(images)):
    print("[Image #%d]" % (i,))
    keypoints_before = KeypointsOnImage.from_xy_array(
        keypoints[i], shape=images[i].shape)
    keypoints_after = KeypointsOnImage.from_xy_array(
        keypoints_aug[i], shape=images_aug[i].shape)
    image_before = keypoints_before.draw_on_image(images[i])
    image_after = keypoints_after.draw_on_image(images_aug[i])
    ia.imshow(np.hstack([image_before, image_after]))

    kps_zipped = zip(keypoints_before.keypoints,
                     keypoints_after.keypoints)
    for keypoint_before, keypoint_after in kps_zipped:
        x_before, y_before = keypoint_before.x, keypoint_before.y
        x_after, y_after = keypoint_after.x, keypoint_after.y
        print("before aug: x=%d y=%d | after aug: x=%d y=%d" % (
            x_before, y_before, x_after, y_after))


    def generator(self, features, labels, batch_size):
        batch_features = np.zeros((batch_size, 128, 128, 3))
        batch_labels = np.zeros((batch_size, 128, 128, 1))
        while True:
            for i in range(batch_size):
                index = random.randint(0, len(features)-1)
                random_augmented_image, random_augmented_labels = self.do_augmentation(self.seq_det, features[index], labels[index])
                batch_features[i] = random_augmented_image
                batch_labels[i] = random_augmented_labels
            yield batch_features, batch_labels

    def do_augmentation(self, seq_det, x_train, y_train):
        ret_y_train = np.zeros((128,128,1))
        ret_y_train[:,:,:1] = y_train

        aug_x_train = seq_det.augment_images([x_train])[0]
        aug_y_train = seq_det.augment_images([ret_y_train])[0]

        ret_x_train = aug_x_train
        ret_y_train = aug_y_train
        return ret_x_train, ret_y_train[:,:,:1]