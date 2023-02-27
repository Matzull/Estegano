#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>
#include <sycl/sycl.hpp>

#include "io_routines.h"
#include "stegano_routines.h"

#define M_PI 3.14159265358979323846 /* pi */

float clamp(float value, float min, float max)
{
	return (value < min) ? min : (value > max) ? max : value;
}



void im2ycbcr(
				uint8_t* _imgIn,
				float* _imgX,
				float* _imgY,
				float* _imgZ,
				int width, int height,
				sycl::queue& Q
			)
{      
	{
		Q.submit([&](sycl::handler &h){
			h.parallel_for(sycl::range<1>(height * width), [=](sycl::id<1> item){
				int t_i = item + item + item;
				_imgX[item] = 0.299 * _imgIn[t_i] + 0.587 * _imgIn[t_i + 1] + 0.114 * _imgIn[t_i + 2];
				_imgY[item] = 128.0 + 0.5 * _imgIn[t_i] - 0.418688 * _imgIn[t_i + 1] - 0.081312 * _imgIn[t_i + 2];
				_imgZ[item] = 128.0 - 0.168736 * _imgIn[t_i] - 0.3331264 * _imgIn[t_i + 1] + 0.5 * _imgIn[t_i + 2];
			});
		}).wait();
	};
}

void ycbcr2im(
				uint8_t* _imgOut,
				float* _imgX,
				float* _imgY,
				float* _imgZ,
				int width, int height,
				sycl::queue& Q
			)
{
	{
		Q.submit([&](sycl::handler &h){

			h.parallel_for(sycl::range<1>(width * height), [=](sycl::id<1> item){
				auto t_i = item + item + item;
				
				_imgOut[t_i] 	 = clamp(_imgX[item] + 1.402 * (_imgY[item] - 128.0), 0, 255);
				_imgOut[t_i + 1] = clamp(_imgX[item] - 0.34414 * (_imgZ[item] - 128.0) - 0.71414 * (_imgY[item] - 128.0), 0, 255);
				_imgOut[t_i + 2] = clamp(_imgX[item] + 1.772 * (_imgZ[item] - 128.0), 0, 255);
			});
		}).wait();
	};
}

void get_dct8x8_params(float* _mcosine, float* _alpha, sycl::queue& Q)
{
	int bM = 8;
	int bN = 8;
	{
		Q.submit([&](sycl::handler &h){

			h.single_task([=](){
				for (int i = 0; i < bM; i++)
				{
					for (int j = 0; j < bN; j++)
					{
						_mcosine[i * bN + j] = cos(((2 * i + 1) * M_PI * j) / (2 * bM));
					}
				}
				_alpha[0] = 1 / sqrt(bM * 1.0f);
				for (int i = 1; i < bM; i++)
				{
					_alpha[i] = sqrt(2.0f) / sqrt(bM * 1.0f);
				}
			});
		}).wait();
	};
	
}

// function for DCT. Picture divide block size 8x8
void dct8x8_2d(
				float* _in,
				float* _out,
				int width, int height,
				float* _mcosine,
				float* _alpha,
				sycl::queue& Q
				)
{
	const int bM = 8;
	const int bN = 8;

	{
		Q.submit([&](sycl::handler &h){

			h.parallel_for(sycl::range<2>(height / bM, width / bN), [=](sycl::id<2> item){
				auto stride_i = int(item[0] * bM);//y axis
				auto stride_j = int(item[1] * bN);//x axis
				for (int i = 0; i < bM; i++)
				{
					for (int j = 0; j < bN; j++)
					{
						float tmp = 0.0;
						for (int ii = 0; ii < bM; ii++)
						{
							for (int jj = 0; jj < bN; jj++)
							{
								tmp += _in[(stride_i + ii) * width + stride_j + jj] * _mcosine[ii * bN + i] * _mcosine[jj * bN + j];
							}
						}
						_out[(stride_i + i) * width + stride_j + j] = tmp * _alpha[i] * _alpha[j];
					}
				}		
			});
		}).wait();
	};
}

void idct8x8_2d(
				float* _in,
				float* _out,
				int width, int height,
				float* _mcosine,
				float* _alpha,
				sycl::queue& Q
				)
{
	int bM = 8;
	int bN = 8;

	{
		Q.submit([&](sycl::handler &h){

			h.parallel_for(sycl::range<2>(height / bN, width / bM), [=](sycl::id<2> item){
				int stride_i = item[0] * bM;
				int stride_j = item[1] * bN;
				for (int i = 0; i < bM; i++)
				{
					for (int j = 0; j < bN; j++)
					{
						float tmp = 0.0;
						for (int ii = 0; ii < bM; ii++)
						{
							for (int jj = 0; jj < bN; jj++)
								tmp += _in[(stride_i + ii) * width + stride_j + jj] * _mcosine[i * bN + ii] * _mcosine[j * bN + jj] * _alpha[ii] * _alpha[jj];
						}
						_out[(stride_i + i) * width + stride_j + j] = tmp;
					}
				}	
			});
		}).wait();
	};
}

void dct8x8_2d(float *in, float *out, int width, int height, float *mcosine, float *alpha)
{
	int bM=8;
	int bN=8;

	for(int bi=0; bi<height/bM; bi++)
	{
		int stride_i = bi * bM;
		for(int bj=0; bj<width/bN; bj++)
		{
			int stride_j = bj * bN;
			for (int i=0; i<bM; i++)
			{
				for (int j=0; j<bN; j++)
				{
					float tmp = 0.0;
					for (int ii=0; ii < bM; ii++) 
					{
						for (int jj=0; jj < bN; jj++)
							tmp += in[(stride_i+ii)*width + stride_j+jj] * mcosine[ii*bN+i]*mcosine[jj*bN+j];
					}
					out[(stride_i+i)*width + stride_j+j] = tmp*alpha[i]*alpha[j];
				}
			}
		}
	}
}

void idct8x8_2d(float *in, float *out, int width, int height, float *mcosine, float *alpha)
{
	int bM=8;
	int bN=8;

	for(int bi=0; bi<height/bM; bi++)
	{
		int stride_i = bi * bM;
		for(int bj=0; bj<width/bN; bj++)
		{
			int stride_j = bj * bN;
			for (int i=0; i<bM; i++)
			{
				for (int j=0; j<bN; j++)
				{
					float tmp = 0.0;
					for (int ii=0; ii < bM; ii++) 
					{
						for (int jj=0; jj < bN; jj++)
							tmp += in[(stride_i+ii)*width + stride_j+jj] * mcosine[i*bN+ii]*mcosine[j*bN+jj]*alpha[ii]*alpha[jj];
					}
					out[(stride_i+i)*width + stride_j+j] = tmp;
				}
			}
		}
	}
}

void insert_msg(
				float* _img,
				int width, int height, 
				char* _msg,
				int msg_length,
				sycl::queue& Q
				)
{
	{
		Q.submit([&](sycl::handler &h){
			int i_insert = 3;
			int j_insert = 4;

			int bM = 8;
			int bN = 8;

			int bsI = height / bM;
			int bsJ = width / bN;

			if (bsI * bsJ < msg_length * 8)
				printf("Image not enough to save message!!!\n");

			h.parallel_for(sycl::range<2>(msg_length, 8), [=](sycl::id<2> item){
				auto c = item[0];
				auto b = item[1];

				int bi = (c * 8 + b) / bsJ;
				int bj = (c * 8 + b) % bsJ;
				
				char ch = _msg[c];
				char bit = (ch & (1 << b)) >> b;

				int stride_i = bi * bM;
				int stride_j = bj * bN;
				float tmp = 0.0;
				for (int ii = 0; ii < bM; ii++)
				{
					for (int jj = 0; jj < bN; jj++)
						tmp += _img[(stride_i + ii) * width + stride_j + jj];
				}
				float mean = tmp / (bM * bN);

				if (bit)
					_img[(stride_i + i_insert) * width + stride_j + j_insert] = fabsf(mean); //+
				else
					_img[(stride_i + i_insert) * width + stride_j + j_insert] = -1.0f * fabsf(mean); //-
			});
		}).wait();	
	};
}

/**/
void insert_msg(float *img, int width, int height, char *msg, int msg_length)
{
	int i_insert=3;
	int j_insert=4;

	int bM=8;
	int bN=8;
		
	int bsI = height/bM;
	int bsJ = width/bN;
	int bi = 0;
	int bj = 0;
	
	if(bsI*bsJ<msg_length*8)
		printf("Image not enough to save message!!!\n");

	for(int c=0; c<msg_length; c++)
		for(int b=0; b<8; b++)
		{
			char ch = msg[c];
			char bit = (ch&(1<<b))>>b;
			
			int stride_i = bi * bM;
			int stride_j = bj * bN;
			float tmp = 0.0;
			for (int ii=0; ii < bM; ii++) 
			{
				for (int jj=0; jj < bN; jj++)
					tmp += img[(stride_i+ii)*width + stride_j+jj];
			}
			float mean = tmp/(bM*bN);
			
//			img[(bi+i_insert)*width + bj+j_insert] = (float)(bit)*img[(bi+i_insert)*width + bj+j_insert];

			if (bit) 
				img[(stride_i+i_insert)*width + stride_j+j_insert] = fabsf(mean); //+
			else
				img[(stride_i+i_insert)*width + stride_j+j_insert] = -1.0f*fabsf(mean); //-


			bj++;
			if (bj>=bsJ){
				bj=0;
				bi++;
			}
		}
}
/**/

void extract_msg(
					float* _img,
					int width, int height, 
					char* _msg,
					int msg_length,
					sycl::queue& Q
				)
{
	{
		Q.submit([&](sycl::handler &h){

			int i_insert = 3;
			int j_insert = 4;

			int bM = 8;
			int bN = 8;

			int bsJ = width / bN;
			
			h.parallel_for(sycl::range<1>(msg_length), [=](sycl::id<1> item){
				char ch = 0;
				int c = (int)item;

				for (int b = 0; b < 8; b++)
				{
					int bi = (c * 8 + b) / bsJ;
					int bj = (c * 8 + b) % bsJ;
					int bit;

					int stride_i = bi * bM;
					int stride_j = bj * bN;
					float tmp = 0.0;
					for (int ii = 0; ii < bM; ii++)
					{
						for (int jj = 0; jj < bN; jj++)
							tmp += _img[(stride_i + ii) * width + stride_j + jj];
					}
					float mean = tmp / (bM * bN);

					if (_img[(stride_i + i_insert) * width + stride_j + j_insert] > 0.5f * mean)
						bit = 1;
					else
						bit = 0;

					ch = (bit << b) | ch;

				}
				_msg[c] = ch;
			});
		}).wait();	
	};
}


void encoder(char *file_in, char *file_out, char *c_msg, int msg_len, sycl::queue &Q)
{

	int w, h;
	// im corresponds to R0G0B0,R1G1B1.....
	uint8_t *im = (uint8_t *)loadPNG(file_in, &w, &h);
	uint8_t *im_out = (uint8_t *)malloc(3 * w * h);
	{

		uint8_t* _imgIn = sycl::malloc_device<uint8_t>(3 * w * h, Q);//memset
		float* _imgX = sycl::malloc_device<float>(w * h, Q);
		float* _imgY = sycl::malloc_device<float>(w * h, Q);
		float* _imgZ = sycl::malloc_device<float>(w * h, Q);
		uint8_t* _imgOut = sycl::malloc_device<uint8_t>(3 * w * h, Q);
		float* _Ydct = sycl::malloc_device<float>(w * h, Q);
		float* _mcosine = sycl::malloc_device<float>(8 * 8, Q);
		float* _alpha = sycl::malloc_device<float>(8, Q);
		char* _msg = sycl::malloc_device<char>(msg_len, Q);

		Q.memcpy(_imgIn, im, (3 * w * h));
		Q.memcpy(_msg, c_msg, msg_len);
		
		// Create imRGB & imYCrCb
		
		get_dct8x8_params(_mcosine, _alpha, Q);
		
		double start = omp_get_wtime();

		im2ycbcr(_imgIn, _imgX, _imgY, _imgZ, w, h, Q);

		dct8x8_2d(_imgX, _Ydct, w, h, _mcosine, _alpha, Q);

		// Insert Message
		//insert_msg(_Ydct, h, w, _msg, msg_len, Q);

		float* Ydct = (float*)malloc(w * h * sizeof(float));
		float* imgX = (float*)malloc(w * h * sizeof(float));
		float* mcosine = (float*)malloc(8 * 8 * sizeof(float));
		float* alpha = (float*)malloc(w * h * sizeof(float));

		Q.memcpy(imgX, _imgX, w * h * sizeof(float));
		Q.memcpy(mcosine, _mcosine, 8 * 8 * sizeof(float));
		Q.memcpy(alpha, _alpha, 8 * sizeof(float));
		Q.memcpy(Ydct, _Ydct, w * h * sizeof(float));
		insert_msg(Ydct, h, w, c_msg, msg_len);
		idct8x8_2d(Ydct, imgX, w, h, mcosine, alpha);
		Q.memcpy(_imgX, imgX, w * h * sizeof(float));

		//idct8x8_2d(_Ydct, _imgX, w, h, _mcosine, _alpha, Q);	

		ycbcr2im(_imgOut, _imgX, _imgY, _imgZ, w, h, Q);

		double stop = omp_get_wtime();
		printf("Encoding time=%f sec.\n", stop - start);

		Q.memcpy(im_out, _imgOut, 3 * (w * h));

		savePNG(file_out, im_out, w, h);

		sycl::free(_imgIn, Q);
		sycl::free(_imgOut, Q);
		sycl::free(_imgX, Q);
		sycl::free(_imgY, Q);
		sycl::free(_imgZ, Q);
		sycl::free(_Ydct, Q);
		sycl::free(_mcosine, Q);
		sycl::free(_alpha, Q);
		sycl::free(_msg, Q);
	};
}

void decoder(char *file_in, char *msg_decoded, int msg_len, sycl::queue &Q)
{

	int w, h;

	uint8_t *im = (uint8_t *)loadPNG(file_in, &w, &h);

	{
		
		uint8_t* _imgIn = sycl::malloc_device<uint8_t>(3 * w * h, Q);//memset
		float* _imgX = sycl::malloc_device<float>(w * h, Q);
		float* _imgY = sycl::malloc_device<float>(w * h, Q);
		float* _imgZ = sycl::malloc_device<float>(w * h, Q);
		float* _Ydct = sycl::malloc_device<float>(w * h, Q);
		float* _mcosine = sycl::malloc_device<float>(8 * 8, Q);
		float* _alpha = sycl::malloc_device<float>(8, Q);
		char* _msg = sycl::malloc_device<char>(msg_len, Q);

		Q.memcpy(_imgIn, im, 3 * (w * h));

		// Create imRGB & imYCrCb

		get_dct8x8_params(_mcosine, _alpha, Q);

		double start = omp_get_wtime();

		im2ycbcr(_imgIn, _imgX, _imgY, _imgZ, w, h, Q);
		
		dct8x8_2d(_imgX, _Ydct, w, h, _mcosine, _alpha, Q);

		extract_msg(_Ydct, w, h, _msg, msg_len, Q);

		double stop = omp_get_wtime();
		printf("Decoding time=%f sec.\n", stop - start);
		Q.memcpy(msg_decoded, _msg, msg_len);

		sycl::free(_imgIn, Q);
		sycl::free(_imgX, Q);
		sycl::free(_imgY, Q);
		sycl::free(_imgZ, Q);
		sycl::free(_Ydct, Q);
		sycl::free(_mcosine, Q);
		sycl::free(_alpha, Q);
		sycl::free(_msg, Q);
	
	};
}
