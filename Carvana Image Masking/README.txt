Fully convolutional network implemented for Kaggle Carvana Masking Challenge:
https://www.kaggle.com/c/carvana-image-masking-challenge

Model inspired by paper:
https://arxiv.org/abs/1605.06211



### LOG ###
170919
First try on EDA. Figured out image reading and vectorisation. Applied averaging on masks.

171013
Numerous attempts on FCN later...FCN_64 (basic) and a properly written FCN_8 with image data augmentation (scale, rotate, translate).

171017
Previous bias was due to faulty image queuer on FloydHub. Added full implementation of FC8 with Xavier init for fc layers and Bilinear init for deconv filters. Satisfactory result.
Final train and valid loss ~ 0.04 using (softmax (binary) + dice)

### END ###


Further works:
FCN is quite similar and can be adapted to become Segnet (discard fc layers and reverse the downsampling completely):
https://arxiv.org/abs/1511.00561

The idea of skip layers highly resembles to that implemented in Unet:
https://arxiv.org/abs/1505.04597

Further tuning of optimiser hyperparameters can be done especially on the momentum. A more progressive learning rate decay is also recommended.

Data augmentation should also have more variation.
