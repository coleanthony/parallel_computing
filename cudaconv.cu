#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#pragma pack(1)

typedef struct BITMAPFILEHEADER {
	unsigned short int  bfType;
	unsigned int bfSize;
	unsigned short int  bfReserved1;
	unsigned short int  bfReserved2;
	unsigned int bfOffBits;
} BITMAPFILEHEADER;

typedef struct BITMAPINFOHEADER {
	unsigned int biSize;
	unsigned int biWidth;
	unsigned int biHeight;
	unsigned short int  biPlanes;
	unsigned short int  biBitCount;
	unsigned int biCompression;
	unsigned int biSizeImage;
	unsigned int biXPelsPerMeter;
	unsigned int biYPelsPerMeter;
	unsigned int biClrUsed;
	unsigned int biClrImportant;
} BITMAPINFOHEADER;


static void HandleError(cudaError_t err, const char *file, int line){
	if (err != cudaSuccess){
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

//给定一个图像位图数据、宽、高、颜色表指针及每像素所占的位数等信息,将其写到指定文件中
void savebmpfile(char *bmpName, unsigned char *imgBuf, int width, int height, int biBitCount) {
	int colorTablesize = 0;

	if (biBitCount == 8)
		colorTablesize = 1024;  // 8*128

	//待存储图像数据每行字节数为4的倍数
	int lineByte = (width * biBitCount / 8 + 3) / 4 * 4;
	//以二进制写的方式打开文件
	FILE *fp = fopen(bmpName, "wb");
	//申请位图文件头结构变量，填写文件头信息
	BITMAPFILEHEADER fileHead;

	fileHead.bfType = 0x4D42;  // bmp类型
	// bfSize是图像文件4个组成部分之和
	fileHead.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + colorTablesize + lineByte * height;
	fileHead.bfReserved1 = 0;
	fileHead.bfReserved2 = 0;

	// bfOffBits是图像文件前3个部分所需空间之和
	fileHead.bfOffBits = 54 + colorTablesize;

	//写文件头进文件
	fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);

	//申请位图信息头结构变量，填写信息头信息
	BITMAPINFOHEADER head;

	head.biBitCount = biBitCount;
	head.biClrImportant = 0;
	head.biClrUsed = 0;
	head.biCompression = 0;
	head.biHeight = height;
	head.biPlanes = 1;
	head.biSize = 40;
	head.biSizeImage = lineByte * height;
	head.biWidth = width;
	head.biXPelsPerMeter = 0;
	head.biYPelsPerMeter = 0;

	fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);
	fwrite(imgBuf, height * lineByte, 1, fp);

	fclose(fp);
}


__global__ void image_convolution_kernel(unsigned char *pBmpBufGPU, unsigned char *resBufGPU,double *kernelGPU,int bmpWidth, int bmpHeight) {
	double accum;
	int col = threadIdx.x + blockIdx.x * blockDim.x;   //col index
	int row = threadIdx.y + blockIdx.y * blockDim.y;   //row index
	int maskRowsRadius = 2;
	int maskColsRadius = 2;

	for (int k = 0; k < 3; k++) {      //cycle on kernel channels
		if (row < bmpHeight && col < bmpWidth) {
			accum = 0.0;
			int startRow = row - maskRowsRadius;  //row index shifted by mask radius
			int startCol = col - maskColsRadius;  //col index shifted by mask radius
			for (int i = 0; i < 5; i++) { //cycle on mask rows
				for (int j = 0; j < 5; j++) { //cycle on mask columns
					int currentRow = startRow + i; // row index to fetch data from input image
					int currentCol = startCol + j; // col index to fetch data from input image
					if (currentRow >= 0 && currentRow < bmpHeight && currentCol >= 0 && currentCol < bmpWidth) 
						accum += pBmpBufGPU[(currentRow * bmpWidth + currentCol)*3 + k] *kernelGPU[i * 5 + j];
					else 
						accum = 0.0;
				}
			}
			resBufGPU[(row* bmpWidth + col) * 3 + k] = accum;
		}
	}
}

int main() {
	char filename[] = "timg.bmp";
	char writePath[] = "result.bmp";
	int kernelsize = 25;
	const int blocksize=16;
	double kernelx[kernelsize] = { 0.01441881,0.02808402,0.0350727, 0.02808402,0.01441881,
		                           0.02808402,0.05470020,0.06831229,0.05470020,0.02808402,
								   0.03507270,0.06831229,0.08531173,0.06831229,0.03507270,
								   0.02808402,0.05470020,0.06831229,0.05470020,0.02808402,
								   0.01441881,0.02808402,0.03507270,0.02808402,0.01441881};

	unsigned char *resBuf = NULL;
	unsigned char *pBmpBuf = NULL;  //读入图像数据的指针

	//read bmp
	int bmpWidth;    //图像的宽
	int bmpHeight;   //图像的高
	int BiBitCount;  //图像类型，每像素位数 8-灰度图 24-彩色图
	
	BITMAPFILEHEADER BmpHead;
	BITMAPINFOHEADER BmpInfo;

	FILE *fp = fopen(filename, "rb");  //二进制读方式打开指定的图像文件

	fread(&BmpHead, sizeof(BITMAPFILEHEADER), 1, fp);
	fread(&BmpInfo, sizeof(BITMAPINFOHEADER), 1, fp);

	bmpWidth = BmpInfo.biWidth;
	bmpHeight = BmpInfo.biHeight;
	BiBitCount = BmpInfo.biBitCount;

	int lineByte = (bmpWidth * BiBitCount / 8 + 3) / 4 * 4;

	pBmpBuf = (unsigned char*)malloc(lineByte * bmpHeight);
	resBuf = (unsigned char*)malloc(bmpWidth * 3 * bmpHeight);

	fread(pBmpBuf, lineByte * bmpHeight, 1, fp);
	printf("read bmp file successfully\n");

	//cuda conv
	unsigned char *pBmpBufGPU;
	unsigned char *resBufGPU;
	double *kernelGPU;

	HANDLE_ERROR(cudaMalloc(&pBmpBufGPU, 3*bmpWidth * bmpHeight * sizeof(unsigned char)));
	HANDLE_ERROR(cudaMalloc(&resBufGPU, 3*bmpWidth * bmpHeight * sizeof(unsigned char)));
	HANDLE_ERROR(cudaMalloc(&kernelGPU, kernelsize * sizeof(double)));
	HANDLE_ERROR(cudaMemcpy(pBmpBufGPU, pBmpBuf, 3 * bmpWidth * bmpHeight * sizeof(unsigned char), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(kernelGPU, kernelx, kernelsize * sizeof(double), cudaMemcpyHostToDevice));

	dim3 dimBlock(blocksize,blocksize,1);
	dim3 dimGrid((bmpWidth+blocksize-1)/blocksize,(bmpHeight+blocksize-1)/blocksize);
	
	printf("cuda global memory convolution\n");
	printf("image dimensions %d %d\n", bmpWidth, bmpHeight);
	image_convolution_kernel <<<dimGrid,dimBlock>>> (pBmpBufGPU, resBufGPU, kernelGPU,bmpWidth, bmpHeight);

	HANDLE_ERROR(cudaMemcpy(resBuf, resBufGPU, 3*bmpWidth * bmpHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	printf("convolution finished\n");
	
	savebmpfile(writePath, resBuf, bmpWidth, bmpHeight, BiBitCount);
	printf("Save bmp file succussfully\n");
	if (pBmpBuf) free(pBmpBuf);
	if (resBuf) free(resBuf);

	cudaFree(pBmpBufGPU);
	cudaFree(resBufGPU);

	return 0;
}
