#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>
#include <fstream>
#include <complex>

using namespace cv;
using namespace std;

double deg_2_rad = M_PI / 180;


#define SIGMA_CLIP 6.0f
inline int sigma2radius(float sigma);
inline float radius2sigma(int r);
void motionBlur(Mat& input, Mat& output, int weight);
void createMotionFilterMask(Size imsize, Mat& mask, int weight);
void fftShift(Mat magI);
void computeDFT(Mat& image, Mat& dest);
void createGaussFilterMask(Size imsize, Mat &mask, int radius);
void deconvoluteWiener(Mat& img, Mat& kernel,float snr);
void myMedianFilter(Mat &image, int size);
double MSE(Mat original, Mat output, int channel);
double PSNR(Mat original, Mat output);


int main(int argc, char ** argv)
{
    if(argc < 3)
        cout << "./executable [input] [answer]" << endl;


    char* filename = argv[1];
    char* outFilename;

    bool median = false;

    int r = 0;
    int snr = 0;
    int weight = 0;
    int choice = 0;
    for(char* it = filename; *it; ++it) 
        if(*it == '3'){
            choice = 3;
            median = true;
            r = 23;
            snr = 28;
            outFilename = "output3.bmp";
        }
        else if(*it == '2'){
            choice = 2;
            weight = 10;
            snr = 33;
            outFilename = "output2.bmp";
        }
        else if(*it == '1'){
            choice = 1;
            r = 12;
            snr = 45;
            outFilename = "output1.bmp";
        }


    Mat inputImage = imread(filename);
    if( inputImage.empty())
        return -1;
    
    /*Prepare container for final image output*/
    Mat mergedImage = inputImage.clone();

    
    /*Display answer for comparing purpose*/
    Mat answer = imread(argv[2]);
    if(!answer.empty())
        imshow("answer", answer);

    imshow("input", inputImage);
    if(median){
        myMedianFilter(inputImage, 7);
        imwrite("median.jpg", inputImage);
    }

    // imshow("med fil", inputImage);

    Mat channel[3];
    split(inputImage, channel);

    Mat I;
    Mat outputChannel[3];

    // string wname = "blur test";
    // namedWindow(wname);


    // int r = 20; createTrackbar("r",wname,&r,100);
    // int snr = 0; createTrackbar("snr:Weiner",wname, &snr,30);//parameter for Weiner filter
    // int weight = 10; createTrackbar("w",wname,&weight,30);
    // int m_g = 0; createTrackbar("motion/Gaussian",wname,&m_g,1);
    int m_g = 0;
    if(choice == 2)
        m_g = 0;
    else
        m_g = 1;


    int key = 0;

    // r -= 7;
    // snr -= 10;
    // weight -=7;
    // double maxPSNR = 0;
    // double maxr = 0;
    // double maxSNR = 0;
    // double maxWeight = 0;
    // for(int count1 = 0; count1 < 15; count1++){
            // r += 1;
            // weight += 1;
    // for(int count2 = 0; count2 < 20; count2++){
            // snr += 1;
    while(key!='q'){
    for (int i = 0; i < 3; i++){

        /*Work on a single channel*/
        I = channel[i].clone();

        /*FFT*/
        Mat dftMat;
        computeDFT(I, dftMat);

        /*Create Mask*/
        Mat mask;
        if(!m_g)
            createMotionFilterMask(dftMat.size(), mask, weight);
        else
            createGaussFilterMask(dftMat.size(), mask, r);

        fftShift(mask);

        /*Obtain frequency domain of Gaussian Mask*/
        Mat kernel;
        computeDFT(mask,kernel);

        /*Wiener Filter to fix gaussian Blur*/
        deconvoluteWiener(dftMat, kernel, snr);

        /*Inverse DFT to Spacial domain*/
        Mat wienerInverseTransform;
        idft(dftMat, wienerInverseTransform,DFT_REAL_OUTPUT+DFT_SCALE);


        /*Store results in temporary channel*/
        cv::Mat finalImage;
        wienerInverseTransform.convertTo(finalImage, CV_8U);
        outputChannel[i] = finalImage.clone();

    }


    Mat outPlane[] = {outputChannel[0], outputChannel[1], outputChannel[2]};
    merge(outPlane, 3, mergedImage);

    imshow("result", mergedImage);
    key = waitKey(1);
    }


    imwrite(outFilename, mergedImage);


    // if (PSNR(answer, mergedImage) > maxPSNR){
        // maxPSNR = PSNR(answer, mergedImage);
        // maxr = r;
        // maxSNR = snr;
        // maxWeight = weight;
        // 
        // cout << "current:" << endl;
        // cout << "maxPSNR: " << maxPSNR << endl;
        // cout << "maxr: " << maxr << endl;
        // cout << "maxWeight" << maxWeight <<endl;
        // cout << "maxSNR: " << maxSNR << endl;
        // cout << "=================" << endl << endl;
    // }
    // destroyAllWindows();
    
    // }
        // snr -= 20; 
    // }

    // cout << "RESULT" << endl;
    // cout << "maxPSNR: " << maxPSNR << endl;
    // cout << "maxr: " << maxr << endl;
    // cout << "maxWeight" << maxWeight <<endl;
    // cout << "maxSNR: " << maxSNR << endl;
    // cout << "=================" << endl << endl;

    cout << "PSNR: " << PSNR(answer, mergedImage) << endl;

    return 0;
}

/*Convert sigma to radius for Gaussian Blur*/
inline int sigma2radius(float sigma)
{
    return (int)(SIGMA_CLIP*sigma+0.5f);
}

inline float radius2sigma(int r)
{
    return (r/SIGMA_CLIP+0.5f);
}


/*Motion Blur*/
void motionBlur(Mat& input, Mat& output, int weight){

    int size = weight*2 - 1;
    vector<vector< double> > kernel;
    kernel.resize(size, vector<double>(size,0));
    // cout << size << endl;
    

    for(int i = 0; i < size; i++)
        for(int j = 0; j < size; j++)
            if((i==(size - j - 1)) && (i>=weight-1) ) //diagonal
                kernel[i][j] = (1.0 / weight);
            else
                kernel[i][j] = 0;


    for(int row = 0; row < output.rows; row++)
        for(int col = 0; col < output.cols; col++){
            for(int pixel = 0; pixel < 3; pixel++){
                int sum = 0;
                for(int kernel_x = -(weight-1); kernel_x <= (weight-1); kernel_x++){
                    for(int kernel_y = -(weight - 1); kernel_y <= (weight - 1); kernel_y++){
                        int currentCol = col + kernel_x;
                        int currentRow = row + kernel_y;
                        if(currentCol<0 || currentRow<0 || currentCol>=output.cols || currentRow>=output.rows)
                            sum += input.at<Vec3b>(row, col).val[pixel] * kernel[kernel_y+(weight-1)][kernel_x+(weight-1)];
                        else
                            sum += input.at<Vec3b>(currentRow, currentCol).val[pixel] * kernel[kernel_y+(weight-1)][kernel_x+(weight-1)];
                    }

                }
                output.at<Vec3b>(row, col).val[pixel] = sum;
            }
        } 
}


double PSNR(Mat original, Mat output){
    double sum = 0;
    for(int i = 0; i < 3; i++)
        sum += 10 * log10(pow(255,2) / MSE(original,output,i));

    return sum;
}

double MSE(Mat original, Mat output, int channel){
    int rows = original.rows;
    int cols = original.cols;
    double sum = 0;
    for(int row = 0; row < rows; row++)
        for(int col = 0; col < cols; col++)
            sum += pow(output.at<Vec3b>(row,col).val[channel] - original.at<Vec3b>(row,col).val[channel], 2);

    return sum / (rows*cols); 
}

/*Create motion blur mask*/
void createMotionFilterMask(Size imsize, Mat& mask, int weight){

    int size = weight*2 - 1;
    Mat kernel(size,size,CV_32F,Scalar::all(0));
    // vector<vector< double> > kernel;
    // kernel.resize(size, vector<double>(size,0));
    // cout << size << endl;
    

    for(int i = 0; i < size; i++)
        for(int j = 0; j < size; j++)
            if((i==(size - j - 1)) && (i>=weight-1) ) //diagonal
                kernel.at<float>(i,j) = (1.0 / weight);
            else
                kernel.at<float>(i,j) = 0;


    int w = imsize.width-kernel.cols;
    int h = imsize.height-kernel.rows;

    int r = w/2;
    int l = imsize.width-kernel.cols -r;

    int b = h/2;
    int t = imsize.height-kernel.rows -b;

    // Mat ret;
    copyMakeBorder(kernel,mask,t,b,l,r,BORDER_CONSTANT,Scalar::all(0));
}

/*move quadrants*/
void fftShift(Mat magI)
{
    // crop if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

/*Compute DFT*/
void computeDFT(Mat& image, Mat& dest)
{
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( image.rows );
    int n = getOptimalDFTSize( image.cols ); // on the border add zero values

    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_REPLICATE);

    Mat imgf;
    padded.convertTo(imgf,CV_32F);  
    dft(imgf, dest, DFT_COMPLEX_OUTPUT);  // furier transform

}

/*Create Gaussian Filter*/
void createGaussFilterMask(Size imsize, Mat &mask, int radius)
{
    // call openCV gaussian kernel generator
    double sigma = radius2sigma(radius);
    Mat kernelX = getGaussianKernel(2*radius+1, sigma, CV_32F);
    Mat kernelY = getGaussianKernel(2*radius+1, sigma, CV_32F);
    // create 2d gaus
    Mat kernel = kernelX * kernelY.t();

/*
    for(int i = 0 ; i < kernel.rows; i++)
        for(int j = 0; j < kernel.cols; j++)
            cout << "[" << i <<"][" << j << "] = " << kernel.at<float>(i,j) << endl;*/

    int w = imsize.width-kernel.cols;
    int h = imsize.height-kernel.rows;

    int r = w/2;
    int l = imsize.width-kernel.cols -r;

    int b = h/2;
    int t = imsize.height-kernel.rows -b;

    // Mat ret;
    copyMakeBorder(kernel,mask,t,b,l,r,BORDER_CONSTANT,Scalar::all(0));

    // return ret;
}

/*Wiener Filter*/
void deconvoluteWiener(Mat& img, Mat& kernel,float snr)
{
    int width = img.cols;
    int height=img.rows;

    Mat_<Vec2f> src = kernel;
    Mat_<Vec2f> dst = img;

    float eps =  + 0.0001f;
    float power, factor, tmp;
    float inv_snr = 1.f / (snr + 0.00001f);
    complex<double> snrComplex(inv_snr,0);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            complex<double> imageComplex(img.at<Vec2f>(y,x)[0],img.at<Vec2f>(y,x)[1]);
            complex<double> kernelComplex(kernel.at<Vec2f>(y,x)[0] + eps,kernel.at<Vec2f>(y,x)[1] + eps);
            complex<double> kernel2 = kernelComplex*conj(kernelComplex);
            complex<double> result = imageComplex * conj((kernel2 / kernelComplex) / (kernel2 + snrComplex));
            img.at<Vec2f>(y,x)[0] = result.real();
            img.at<Vec2f>(y,x)[1] = result.imag();


            /*power = src(y,x)[0] * src(y,x)[0] + src(y,x)[1] * src(y,x)[1]+eps;
            factor = (1.f / power)*(1.f-inv_snr/(power*power + inv_snr));
            // factor = power / (power*power + inv_snr);
            tmp = dst(y,x)[0];
            dst(y,x)[0] = (src(y,x)[0] * tmp + src(y,x)[1] * dst(y,x)[1]) * factor;
            dst(y,x)[1] = (src(y,x)[0] * dst(y,x)[1] - src(y,x)[1] * tmp) * factor; */
        }
    }
}

/*Median Filter*/
void myMedianFilter(Mat &image, int size){
    Mat output = image.clone();
    int length = size / 2;
    for(int row = 0; row < image.rows; row++)
        for(int col = 0; col < image.cols; col++){
            for(int pixel = 0; pixel < 3; pixel++){
                int sum = 0;
                vector<uchar> sortList;
                for(int kernel_x = -length; kernel_x <= length; kernel_x++){
                    for(int kernel_y = -length; kernel_y <= length; kernel_y++){
                        int currentCol = col + kernel_x;
                        int currentRow = row + kernel_y;
                        if(currentCol<0 || currentRow<0 || currentCol>=output.cols || currentRow>=output.rows)
                            sortList.push_back(image.at<Vec3b>(row, col).val[pixel]);
                        else
                            sortList.push_back(image.at<Vec3b>(currentRow, currentCol).val[pixel]);
                    }

                }
                sort(sortList.begin(),sortList.end());
                output.at<Vec3b>(row, col).val[pixel] = sortList[(size*size)/2];
            }

        }
    image = output.clone();
}

