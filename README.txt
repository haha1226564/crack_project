Crack Detection Project


Project overview:

Environment: Ubuntu 16.0.4
Train time: validation accuracy 94% with 60 minutes training on GTX 980m

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

Note: I train and validate my model using the on-line dataset. I did NOT train or
validate my model on the dataset you provided. I usd the dataset you provided as a
one time testing data set. I only test my model on the data set you provided one
time in order to test the generalization of my model.

Future works:
I can add fully connected layer of large size i.e. Dense(1024) instead of Dense(512)
to improve model complexity and in turn to improve the validation accuracy. I can also
increase the training time and use strong regulation function to increase the validation 
accuracy.Since the crack is continuous in some sense, I can also improve my model by 
using structured prediction method.

I used convolution neural network approach for this crack detection task. I think traditional
image processing based approach is also very suitable for this task. Specifically,
we can use Morphological Operations to easily detect the crack with high accuracy. Then I can
use model ensemble technique to dramatically improve the accuracy.









