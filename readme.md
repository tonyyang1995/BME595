# Title
image-to-image colorization
## Team members
Ziyang Tang
tang385@purdue.edu
## Goals
Given a grayscale image, colorize it into a realistic RGB image.
## Challenges
1. How to collect dataset
2. How to colorize the grayscale image.
3. If possible, apply GAN and make it into an supervised training model
4. If possible, apply an images sequence and colorize the gray-scale video.

## Restrictions
1. Since this is a term project, I will not train a large model with a huge dataset. I decide to begin with a small dataset which contains a few classes. The data for each classes can be found from the sub-classes from the imagenet. The label can be the original colorful images, and the data can be the converted gray-scale images. 
2. Due to the limited hardware, I will try to use a small network to train the dataset. Probably using the Alexnet or the MobileNet. 