#include <math.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;

double compare_mse(const Mat& p1, const Mat& p2) {
    Mat s1;
    absdiff(p1, p2, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);
    Scalar s = sum(s1);
    double sse = s.val[0] + s.val[1] + s.val[2];
    if (sse <= 1e-10)
        return 0;
    else {
        double mse = sse / (double)(p1.channels() * p1.total());
        return mse;
    }
}

inline double sim_rate(const double mse) {
    return 100 - sqrt(mse)/2.55;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        printf("This program use OpenCV to make a gaussian blurred image with r=5, sigma=1.5 gaussian kernel, and check with your image.\n");
        printf("Usage: ./test <Teacher Given Image Path> <Your Image Path>\n");
        return -1;
    }
    Mat rawImage, blurImage, yourImage;
    rawImage = imread(argv[1], 1);
    yourImage = imread(argv[2], 1);
    if (!rawImage.data)
    {
        printf("No raw image data\n");
        return -1;
    }
    if (!yourImage.data)
    {
        printf("No your image data\n");
        return -1;
    }
    
    float kernel[5][5] = {
    {0.01441881, 0.02808402, 0.03507270, 0.02808402, 0.01441881},
    {0.02808402, 0.05470020, 0.06831229, 0.05470020, 0.02808402},
    {0.03507270, 0.06831229, 0.08531173, 0.06831229, 0.03507270},
    {0.02808402, 0.05470020, 0.06831229, 0.05470020, 0.02808402},
    {0.01441881, 0.02808402, 0.03507270, 0.02808402, 0.01441881}
    };

    Mat k(5, 5, CV_32F, kernel);
    filter2D(rawImage, blurImage, -1, k, Point(-1, -1), 0, BORDER_CONSTANT);

    if (blurImage.size() == yourImage.size() && blurImage.channels() == yourImage.channels()) {
        printf("[PASS] Shape test passed!\n");
    } else {
        printf("[FAIL] Shape test failed.\n");
        printf("Standard image shape:\nheight: %d, width: %d, channel: %d\n", blurImage.size().height, blurImage.size().width, blurImage.channels());
        printf("Your image shape:\nheight: %d, width: %d, channel: %d\n", yourImage.size().height, yourImage.size().width, yourImage.channels());
        return -1;
    }

    double mse = compare_mse(blurImage, yourImage);
    double sim = sim_rate(mse);

    if (sim > 99) {
        printf("[PASS] Similarity test passed!\n");
        printf("MSE result: %.2f\n", mse);
        printf("Similarity rate: %.2f%%\n", sim);
    } else {
        printf("[FAIL] Similarity test failed.\n");
        printf("MSE result: %.2f\n", mse);
        printf("Similarity rate: %.2f%%, less than 99%%\n.", sim);
    }
    return 0;
}