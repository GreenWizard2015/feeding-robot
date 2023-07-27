> This project has been frozen for a long time, and I am unlikely to continue working on this code. However, I am determined to eventually implement the project. Please also note that this is a rough translation into English, and some links may not work due to translation issues.

# Robot Label Recognition

## [Commit #1](https://github.com/GreenWizard2015/feeding-robot/tree/1b2ec16fc4f90d8788800f3333e78b221a17cdda)

Key points:

- A very primitive network based on the U-net architecture is used. It takes a BGR image (preprocessed with only range normalization to 0..1) of size 224*224 as input, and generates a pixel map for 8 classes/points + 1 map for the "background" as output.

- Localizing the point from the segmentation map is quite challenging (the maximum doesn't always point to the correct point, especially during scaling), so it was decided to find the "center of mass" of the predictions. This approach is also used during the network's training, which increased accuracy and stability. Although the dice loss already considers the target probability distribution, calculating the center of mass helped to remove artifacts (especially in points 7 and 8, which are less visible).

- The dataset was iteratively populated. Initially, 10 frames were annotated, and the network was trained on them. Then, another 10 frames were selected where the network made noticeable mistakes, and the network was trained again. Currently, the dataset contains 91 frames. 67 of them were extracted from the first video file (about 1250 frames). However, the network is capable of self-annotating data with sufficiently high accuracy.

- A primitive but effective data augmentation was used. Of course, some implementation details may be questionable, but it allowed achieving correct predictions even on videos from another camera (not always, but mostly).

Video files and trained models can be [downloaded separately](https://drive.google.com/file/d/1XMTd2z23sf3oe3hz0eZJf5uFK_LfDyfE/).

Video demonstration of the work - [youtube](https://youtu.be/qfuOcrQkL3o).

## Adversarial Loss ([commit](https://github.com/GreenWizard2015/feeding-robot/tree/8351aa58ee9fe39e845e50393654861032b813b3))

The input data for the discriminator became the predicted map for each of the labels (excluding the "non-label"), which significantly simplified the task and achieved an adequate accuracy of the discriminator.

The main network could adapt to the discriminator, so a history of samples was introduced. With a low probability, examples of predictions from the main network were either sent to the history or replaced with older ones from the history. Thus, the discriminator gradually learned more about the details that distinguish desired results (ideal point Gaussian spots) from predicted ones.

Usually, GANs are trained independently without any clear loss describing the task (which leads to various problems). In this case, we already have two main losses (dice and center of mass positioning), so the discriminator should at least not hinder the training process (especially during overfitting, which is common in GANs). To avoid all these problems, the following techniques were used:

- The discriminator was trained only when necessary and not in parallel with the main network. The main network was trained either for N epochs, the number of which gradually increased, or until the discriminator no longer provided a clear signal of network improvement (the generated responses should become more similar to the ground truth... if this doesn't happen for more than 10 epochs, the discriminator might have stopped distinguishing them clearly). This reduces the risk of overfitting the discriminator, and we save computational resources.

- The evaluation of the discriminator is scaled in such a way that it contributes no more than 40% to the loss value. Most of the time, the network optimizes the main loss, receiving additional feedback from the discriminator. Also, the strength of this feedback indirectly depends on the accuracy of the discriminator during training. Although it was not observed, the discriminator may stop distinguishing real samples from generated ones, so the significance of its evaluation decreases.

- If the discriminator starts outputting only one value (due to overfitting, for example), it doesn't worsen the training since the main loss starts providing a more stable signal (it's harder to distinguish 0.123+0.01 from 0.113+0.02 than 0.123+1 from 0.113+1).

Additionally, the test run of the main network had to be abandoned. While it helped identify more successful models, it would have required disabling the discriminator, and the benefit of this was questionable. Moreover, the discriminator is trained on the test set, so it is already included in the discriminator, and thus, the main network is trained on the test set.

## Plans and Ideas

- Automatically annotate more data using the current version of the network. The `CAnchorsDetector.combinedDetections` contains a prototype of this functionality, but it needs to be correctly implemented for training on generated data, taking into account the possibility of errors.

- The robot is a rigid structure, and the possible configurations of label placements are limited by it. Therefore, a model can be trained to determine whether a given label placement is valid. Such a model can be useful not only as an additional loss during recognition but also, for example, in motion planning for the manipulator.
