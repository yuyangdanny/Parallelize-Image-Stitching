# Parallel image stitching

There are some work about image stitching include: seqential/OMP/CUDA version

And here are some CUDA version optimization such as: shared memory/unroll/memory coalescing/padding, and so on.

Some example here: (Check for more details at the file)
``` C++
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
```