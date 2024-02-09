#include <iostream>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <boost/filesystem.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

namespace fs = boost::filesystem;
using namespace std;
using namespace cv;
using namespace std::chrono;

#define THREAD_NUM 8
#define MAX_IMG 50
#define BLOCKWIDTH 32


void padding(const cv::Mat& src, cv::Mat& dst, int blockWidth) {

    int paddingCols = src.cols % blockWidth == 0 ? 0 : (blockWidth - src.cols % blockWidth);
    int paddingRows = src.rows % blockWidth == 0 ? 0 : (blockWidth - src.rows % blockWidth);

    dst.create(src.rows + paddingRows, src.cols + paddingCols, src.type());

    dst.setTo(cv::Scalar::all(0));

    src.copyTo(dst(cv::Rect(0, 0, src.cols, src.rows)));
}

__device__ float2 convert_pt(int pixelX, int pixelY, int w, int h){

    float x = pixelX - (w / 2);
    float y = pixelY - (h / 2);

    float f = -w / 2;
    float r = w;

    float omega = w / 2;
    float z0 = f - sqrt(r * r - omega * omega);
    float zc = (
        2 * z0 + sqrt(4 * z0 * z0 - 4 * (x * x / (f * f) + 1) * (z0 * z0 - r * r))) / (2 * (x * x / (f * f) + 1)
    );

    return make_float2((x * zc / f) + (w / 2), (y * zc / f) + (h / 2));
}

__global__ void cylindricalKernel(
    cv::cuda::PtrStepSz<uchar3> img, cv::cuda::PtrStepSz<uchar3> outimg, int width, int height) {

    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    float2 current_pos = convert_pt(pixelX, pixelY, width, height);
    float current_pos_x = current_pos.x;
    float current_pos_y = current_pos.y;
    int top_left_x = (int)current_pos_x;
    int top_left_y = (int)current_pos_y;

    if(
        top_left_x < 0 ||
        top_left_x > width-2 ||
        top_left_y < 0 ||
        top_left_y > height-2
    )
        return ;

    float dx = current_pos_x - top_left_x;
    float dy = current_pos_y - top_left_y;

    float weight_tl = (1.0 - dx)   *   (1.0 - dy);
    float weight_tr = (dx)         *   (1.0 - dy);
    float weight_bl = (1.0 - dx)   *   (dy);
    float weight_br = (dx)         *   (dy);

    uchar valueR = weight_tl * img(top_left_y       , top_left_x        ).x +
                   weight_tr * img(top_left_y       , top_left_x + 1    ).x +
                   weight_bl * img(top_left_y + 1   , top_left_x        ).x +
                   weight_br * img(top_left_y + 1   , top_left_x + 1    ).x;
    uchar valueG = weight_tl * img(top_left_y       , top_left_x        ).y +
                   weight_tr * img(top_left_y       , top_left_x + 1    ).y +
                   weight_bl * img(top_left_y + 1   , top_left_x        ).y +
                   weight_br * img(top_left_y + 1   , top_left_x + 1    ).y;
    uchar valueB = weight_tl * img(top_left_y       , top_left_x        ).z +
                   weight_tr * img(top_left_y       , top_left_x + 1    ).z +
                   weight_bl * img(top_left_y + 1   , top_left_x        ).z +
                   weight_br * img(top_left_y + 1   , top_left_x + 1    ).z;

    outimg(pixelY, pixelX).x = valueR;
    outimg(pixelY, pixelX).y = valueG;
    outimg(pixelY, pixelX).z = valueB;
}

Mat cylindrical(Mat& img){
    cv::Mat paddedImg;
    padding(img, paddedImg, BLOCKWIDTH);
    cv::cuda::GpuMat tmpimg(paddedImg);
    cv::cuda::GpuMat outimg(paddedImg.size(), paddedImg.type());

    int width = img.cols;
    int height = img.rows;

    Mat result(paddedImg.size(), CV_8UC3);

    dim3 blockDim(BLOCKWIDTH, BLOCKWIDTH);
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y
    );
    cylindricalKernel<<<gridDim, blockDim>>>(tmpimg, outimg, width, height);

    cudaDeviceSynchronize();

    outimg.download(result);

	return result;
}

Mat cropImage(Mat &img)
{
    int top=0;
    int bottom=img.rows;

	for(int i = 0; i < img.cols; i++){
		for(int j = 0; j < img.rows; j++){
			if(img.ptr<Vec3b>(j)[i][0] != 0 && img.ptr<Vec3b>(j)[i][1] != 0 && img.ptr<Vec3b>(j)[i][2] != 0){
				top = max(top,j);
				break;
			}
		}
	}

	for(int i = 0; i < img.cols; i++){
		for(int j = img.rows - 1; j >= 0; j--){
			if(img.ptr<Vec3b>(j)[i][0] != 0 && img.ptr<Vec3b>(j)[i][1] != 0 && img.ptr<Vec3b>(j)[i][2] != 0){
				bottom = min(bottom,j);
				break;
			}
		}
	}

	return img(Range(top,bottom),Range(0,img.cols));

}

void findKeyPoints(Mat img, vector<KeyPoint> &keypoints, Mat &descriptors)
{
    Ptr<SIFT> sift = SIFT::create();
    sift->detectAndCompute(img, noArray(), keypoints, descriptors);
}

void matchKeyPoints(Mat &descriptors1, Mat &descriptors2, vector<DMatch> &matches)
{
    FlannBasedMatcher matcher;
    vector<DMatch> tmp_matches;

    //query descriptor, train descriptor
    matcher.match(descriptors1, descriptors2, tmp_matches);

    double minDist = 100;
    for (int i = 0; i < descriptors1.rows; i++) {
        double dist = tmp_matches[i].distance;
        if(dist < minDist) {
            minDist = dist;
        }
    }

    for (int i = 0; i < descriptors1.rows; i++)
    {
        if (tmp_matches[i].distance < max(5 * minDist, 0.02))
        {
            matches.push_back(tmp_matches[i]);
        }
    }
}

void getHomography(vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<DMatch>& matches, Mat &homography)
{
    if (matches.size() < 4)
    {
        cout << "Not enough matches" << endl;
        return;
    }
    vector<Point2f> points1, points2;
    for (int i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

	Mat Htmp(2,3, CV_64F, Scalar(0));
	Htmp.ptr<double>(0)[0]=1;
	Htmp.ptr<double>(1)[1]=1;

	int matchnum=0;

	for(int i = 0;i < points2.size(); i++){
		double tx,ty;
		tx=points1[i].x-points2[i].x;
		ty=points1[i].y-points2[i].y;
		int sum=0;

		for(int j = 0;j < points2.size(); j++){
			double difx=double(points2[j].x)+tx-double(points1[j].x);
			double dify=double(points2[j].y)+ty-double(points1[j].y);
			if(difx<0){
				difx*=(-1);
			}
			if(dify<0){
				dify*=(-1);
			}
			if(difx+dify<3){ // error < threshold 
				sum++;
			}
		}
		// sum bigger mean this translation matrix is better 
		if(sum>matchnum){
			matchnum=sum;
			Htmp.ptr<double>(0)[2]=tx;
			Htmp.ptr<double>(1)[2]=ty;
		}

	}
	homography = Htmp;
}

__global__ void stitchKernel(const cv::cuda::PtrStepSz<uchar3> src, cv::cuda::PtrStepSz<uchar3> warp, int midline) {
    __shared__ uchar3 shared_memory[32][32];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    shared_memory[local_y][local_x] = src(y, x);
    __syncthreads();
    warp(y, x) = shared_memory[local_y][local_x];

}

void stitch(cv::Mat &src, cv::Mat &warp, int midline) {

    cv::Mat paddedSrc;
    padding(src, paddedSrc, BLOCKWIDTH);
    cv::Mat paddedWarp;
    padding(warp, paddedWarp, BLOCKWIDTH);

    cv::cuda::GpuMat d_src(paddedSrc);
    cv::cuda::GpuMat d_warp(paddedWarp);

    dim3 block(BLOCKWIDTH, BLOCKWIDTH);
    dim3 grid(
        (min(midline, paddedWarp.cols) + block.x - 1) / block.x,
        (paddedWarp.rows + block.y - 1) / block.y
    );

    stitchKernel<<<grid, block>>>(d_src, d_warp, midline);

    d_warp.download(paddedWarp);
    cv::Rect roi(0, 0, warp.cols, warp.rows);
    paddedWarp(roi).copyTo(warp);

}

int main(int argc, char* argv[])
{
    omp_set_num_threads(THREAD_NUM);

    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <folder_name>" << endl;
        return 1;
    }

    string inputFolder = "../../";
    inputFolder += argv[1];

    vector<string> images;
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        if (entry.path().extension() == ".JPG") {
            images.push_back(entry.path().string());
        }
    }
    sort(images.begin(), images.end());
    int numOfImages = images.size();

    Mat imgs[MAX_IMG];
    Mat imgs_color[MAX_IMG];
    vector<KeyPoint> keypoints[MAX_IMG];
    Mat descriptors[MAX_IMG];

    int cylindricalDuration = 0;
    int keypointsDuration = 0;
    int matchDuration = 0;
    int transformDuration = 0;
    int stitchDuration = 0;
    int cropDuration = 0;

    auto allStart = high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < numOfImages; i++) {
        imgs_color[i] = imread(images[i]);
    }

    auto compStart = high_resolution_clock::now();

    auto cylindricalStart = high_resolution_clock::now();

    #pragma omp parallel for default(shared)
    for (int i = 0; i < numOfImages; i++) {
        Mat img = imgs_color[i];

        resize(img, img, Size(img.cols / 3, img.rows / 3));
        copyMakeBorder(img, img, 100, 100, 100, 100, BORDER_CONSTANT);

        img = cylindrical(img);
        imgs_color[i] = img;
        cvtColor(img, img, COLOR_BGR2GRAY);
        imgs[i] = img;
    }

    auto cylindricalStop  = high_resolution_clock::now();

    cylindricalDuration = duration_cast<milliseconds>(cylindricalStop - cylindricalStart).count();

    auto keypointStart = high_resolution_clock::now();

    #pragma omp parallel for default(shared) schedule(dynamic)
    for (int i = 0; i < numOfImages; i++) {
        findKeyPoints(imgs[i], keypoints[i], descriptors[i]);
    }

    auto keypointStop = high_resolution_clock::now();

    keypointsDuration = duration_cast<milliseconds>(keypointStop - keypointStart).count();

    vector<DMatch> matches[MAX_IMG - 1];
    auto matchStart = high_resolution_clock::now();

    #pragma omp parallel for default(shared) schedule(dynamic)
    for (int i = 0; i < numOfImages - 1; i++) {
        matchKeyPoints(descriptors[i], descriptors[i+1], matches[i]);
    }
    auto matchStop = high_resolution_clock::now();

    matchDuration = duration_cast<milliseconds>(matchStop - matchStart).count();

    Mat homographies[MAX_IMG - 1];

    auto transformStart = high_resolution_clock::now();

    #pragma omp parallel for default(shared) schedule(dynamic)
    for (int i = 0; i < numOfImages - 1; i++){
        getHomography(keypoints[i], keypoints[i+1], matches[i], homographies[i]);
    }

    auto transformStop = high_resolution_clock::now();

    transformDuration = duration_cast<milliseconds>(transformStop - transformStart).count();

    Mat result = imgs_color[0];
    int dx = 0;
    int dy = 0;

    auto stitchStart = high_resolution_clock::now();

    for (int i = 0; i < numOfImages - 1; i++){
        homographies[i].ptr<double>(0)[2] += dx;
		homographies[i].ptr<double>(1)[2] += dy;
		dx = homographies[i].ptr<double>(0)[2];
		dy = homographies[i].ptr<double>(1)[2];

        int mRows = max(result.rows, imgs_color[i+1].rows + int(homographies[i].ptr<double>(1)[2]));
		int mCols = imgs_color[i+1].cols + int(homographies[i].ptr<double>(0)[2]);
		int midline = (result.cols + int(homographies[i].ptr<double>(0)[2])) / 2;

        Mat warp = Mat::zeros(mRows, mCols, CV_8UC3);
        warpAffine(imgs_color[i+1], warp, homographies[i], Size(mCols, mRows));
		stitch(result, warp, midline);

        result = warp;
    }

    auto stitchStop = high_resolution_clock::now();

    stitchDuration = duration_cast<milliseconds>(stitchStop - stitchStart).count();

    auto cropStart = high_resolution_clock::now();

    result = cropImage(result);

    auto cropEnd = high_resolution_clock::now();

    cropDuration = duration_cast<milliseconds>(cropEnd - cropStart).count();

    auto compEnd = high_resolution_clock::now();

    imwrite("result.jpg", result);

    auto allEnd = high_resolution_clock::now();

    auto compDuration = duration_cast<milliseconds>(compEnd - compStart).count();
    auto duration = duration_cast<milliseconds>(allEnd - allStart).count();

    cout << "Total Elapsed Time: " << duration << " ms" << endl;
    cout << "IO Time: " << duration - compDuration << " ms" << endl;
    cout << "Computational Time " << compDuration << " ms" << endl;
    cout << "Cylindrical: " << cylindricalDuration << " ms" << endl;
    cout << "Keypoints: " << keypointsDuration << " ms" << endl;
    cout << "Matching: " << matchDuration << " ms" << endl;
    cout << "Transform: " << transformDuration << " ms" << endl;
    cout << "Stitching: " << stitchDuration << " ms" << endl;
    cout << "Cropping " << cropDuration << " ms" << endl;
}