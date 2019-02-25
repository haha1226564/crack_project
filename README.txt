Crack Detection Project


Project overview:

Environment: Ubuntu 16.0.4
Train time: 30 minutes on GTX 980m

I used a pre_trained Xception model as a base convolution model, which is used
for image feature extraction. I striped the top fully connected layers of the
Xception model. Then I add three fully connected layers which is used to detect
the cracks in the image. This technique is called transfer learning, which I used
to speed up my train process.

Training phase:
I split the train process into two phases. In the first part of the train phase,
I set my convolution base (Xception) as non-trainable. I only train the three
fully connected layers in order to prevent large gradient flow to destroy the
convolution base. In the second part of the train phase, I set last convolution
block of Xception as trainable. This phase is also call fine tuning phase, which
is used to achieve better validation accuracy.

Image preprocessing method:
I firstly divide all pixel value by 255 in order to make the input to convolution
neural network to be small number between zero and one. Then, I randomly flip the
image horizontally and vertically. I also rotate the image randomly between 0 and
359 degrees. This technique is call data augmentation, which can be used to improve
the validation loss and validation accuracy for small training data set.












