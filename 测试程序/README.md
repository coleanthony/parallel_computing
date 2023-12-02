# MISC
We have some snippets for doing other cool things:

## Gaussian Kernel Calculator
This is a Python 3 snippet for making 2D gaussian kernel for convolution.

### Requirements 
* `python3`
* `numpy`
* `tensorflow`
* `tensorflow-probability`

### Usage
```
python3 gaussian\_kernel\_calculator.py \<blur radius\> \<standard deviation\>
```

After calculation, a txt file with the generated kernel matrix will be generated for further use. Take a look at `gk\_r3\_dev0.84089642.txt`, it gets the same result as [this sample gaussian matrix on Wikipedia](https://en.wikipedia.org/wiki/Gaussian_blur#Sample_Gaussian_matrix).

## Blurred Image Correction Checker
This is an autochecker for the course on checking the correction of student-made gaussian blurred image. We compare the standard blurred image with student-made ones based on MSE:
1. **Size check**: the height/width/channel between two images should be identical to pass the test.
2. **MSE calculation**: subtract two images, square the result, sum all elements in the result matrix, then divide it by (width\*height\*channel).
3. **Similarity rate calculation**: based on the fact that all-black image and all-white image get the highest MSE and lowest similarity (0%)(we provided an all-black image and an all-white image for you to have a try!), we get the square root of the MSE result, divide it by 255, then use 1 to subtract it. The final result should be more than 99% to pass the test.

### C++ 

#### Requirements
* OpenCV library (`imgcodecs`)

#### Usage
1. Compile the code using
```
g++ test.cpp -o test `pkg-config --cflags --libs opencv`
```
2. Run the code with teacher's unprocessed image and your self-made blurred image.

### Python 3 Snippets

#### Requirements
* `python3`
* `opencv-python`
* `numpy`

#### Usage
Do `python3 test_blur.py [teacher's image] [your image]` to check whether your image could pass the test.

### Max Pooling Image Correction Checker
After blurring with a designated kernel, professor asked us to do 2*2 max pooling. `test_pool.py` is the checker.

#### Requirements
* `python3`
* `opencv-python`
* `numpy`
* `scikit-learn`

#### Usage
Do `python3 test_pool.py [teacher's image] [your image]` to check whether your image could pass the test.
