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
				sycl::buffer<uint8_t, 1>& imgIn, 
				sycl::buffer<float, 1>& imgX, 
				sycl::buffer<float, 1>& imgY, 
				sycl::buffer<float, 1>& imgZ,
				int width, int height,
				sycl::queue& Q
			)
{      
	{
		Q.submit([&](sycl::handler &h){
			sycl::accessor acc_imgIn{imgIn, h, sycl::read_only};
			sycl::accessor acc_out_Y{imgX, h, sycl::write_only};
			sycl::accessor acc_out_Cb{imgY, h, sycl::write_only};
			sycl::accessor acc_out_Cr{imgZ, h, sycl::write_only};
			h.parallel_for(sycl::range<1>(height * width), [=](sycl::id<1> item){
				int t_i = item + item + item;
				acc_out_Y[item] = 0.299 * acc_imgIn[t_i] + 0.587 * acc_imgIn[t_i + 1] + 0.114 * acc_imgIn[t_i + 2];
				acc_out_Cb[item] = 128.0 + 0.5 * acc_imgIn[t_i] - 0.418688 * acc_imgIn[t_i + 1] - 0.081312 * acc_imgIn[t_i + 2];
				acc_out_Cr[item] = 128.0 - 0.168736 * acc_imgIn[t_i] - 0.3331264 * acc_imgIn[t_i + 1] + 0.5 * acc_imgIn[t_i + 2];
			});
		}).wait();
	};
}

void ycbcr2im(	
				sycl::buffer<uint8_t, 1>& imgOut, 
				sycl::buffer<float, 1>& imgX, 
				sycl::buffer<float, 1>& imgY, 
				sycl::buffer<float, 1>& imgZ,
				int width, int height,
				sycl::queue& Q
			)
{
	{
		Q.submit([&](sycl::handler &h){
			sycl::accessor acc_imgOut{imgOut, h, sycl::write_only};
			sycl::accessor acc_out_Y{imgX, h, sycl::read_only};
			sycl::accessor acc_out_Cb{imgY, h, sycl::read_only};
			sycl::accessor acc_out_Cr{imgZ, h, sycl::read_only};

			h.parallel_for(sycl::range<1>(width * height), [=](sycl::id<1> item){
				auto t_i = item + item + item;
				
				acc_imgOut[t_i] 	= clamp(acc_out_Y[item] + 1.402 * (acc_out_Cb[item] - 128.0), 0, 255);
				acc_imgOut[t_i + 1] = clamp(acc_out_Y[item] - 0.34414 * (acc_out_Cr[item] - 128.0) - 0.71414 * (acc_out_Cb[item] - 128.0), 0, 255);
				acc_imgOut[t_i + 2] = clamp(acc_out_Y[item] + 1.772 * (acc_out_Cr[item] - 128.0), 0, 255);
			});
		}).wait();
	};
}

void get_dct8x8_params(sycl::buffer<float, 1>& mcosine, sycl::buffer<float, 1>& alpha, sycl::queue& Q)
{
	int bM = 8;
	int bN = 8;
	{
		Q.submit([&](sycl::handler &h){
			sycl::accessor acc_mcosine{mcosine, h, sycl::read_write};
			sycl::accessor acc_alpha{alpha, h, sycl::read_write};

			h.single_task([=](){
				for (int i = 0; i < bM; i++)
				{
					for (int j = 0; j < bN; j++)
					{
						acc_mcosine[i * bN + j] = cos(((2 * i + 1) * M_PI * j) / (2 * bM));
					}
				}
				acc_alpha[0] = 1 / sqrt(bM * 1.0f);
				for (int i = 1; i < bM; i++)
				{
					acc_alpha[i] = sqrt(2.0f) / sqrt(bM * 1.0f);
				}
			});
		}).wait();
	};
	
}

// function for DCT. Picture divide block size 8x8
void dct8x8_2d(
				sycl::buffer<float, 1>& in,
				sycl::buffer<float, 1>& out,
				int width, int height,
				sycl::buffer<float, 1>& mcosine,
				sycl::buffer<float, 1>& alpha,
				sycl::queue& Q
			)
{
	const int bM = 8;
	const int bN = 8;

	{
		Q.submit([&](sycl::handler &h){
			sycl::accessor acc_in{in, h, sycl::read_write};
			sycl::accessor acc_out{out, h, sycl::read_write};
			sycl::accessor acc_mcosine{mcosine, h, sycl::read_write};
			sycl::accessor acc_alpha{alpha, h, sycl::read_write};
			printf("Line: %d\n", __LINE__);
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
								tmp += acc_in[(stride_i + ii) * width + stride_j + jj] * acc_mcosine[ii * bN + i] * acc_mcosine[jj * bN + j];
							}
						}
						acc_out[(stride_i + i) * width + stride_j + j] = tmp * acc_alpha[i] * acc_alpha[j];
					}
				}		
			});
		}).wait();
	};
}

void idct8x8_2d(
				sycl::buffer<float, 1>& in,
				sycl::buffer<float, 1>& out,
				int width, int height,
				sycl::buffer<float, 1>& mcosine,
				sycl::buffer<float, 1>& alpha,
				sycl::queue& Q
			)
{
	int bM = 8;
	int bN = 8;

	{
		Q.submit([&](sycl::handler &h){

			sycl::accessor acc_in{in, h, sycl::read_write};
			sycl::accessor acc_out{out, h, sycl::read_write};
			sycl::accessor acc_mcosine{mcosine, h, sycl::read_write};
			sycl::accessor acc_alpha{alpha, h, sycl::read_write};

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
								tmp += acc_in[(stride_i + ii) * width + stride_j + jj] * acc_mcosine[i * bN + ii] * acc_mcosine[j * bN + jj] * acc_alpha[ii] * acc_alpha[jj];
						}
						acc_out[(stride_i + i) * width + stride_j + j] = tmp;
					}
				}	
			});
		}).wait();
	};
}

void insert_msg(
				sycl::buffer<float, 1>& img,
				int width, int height, 
				sycl::buffer<char, 1>& msg, 
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
			sycl::accessor acc_img{img, h, sycl::read_write};
			sycl::accessor acc_msg{msg, h, sycl::read_write};

			h.parallel_for(sycl::range<2>(msg_length, 8), [=](sycl::id<2> item){
				auto c = item[0];
				auto b = item[1];

				int bi = 0;
				int bj = 0;
				
				char ch = acc_msg[c];
				char bit = (ch & (1 << b)) >> b;

				int stride_i = bi * bM;
				int stride_j = bj * bN;
				float tmp = 0.0;
				for (int ii = 0; ii < bM; ii++)
				{
					for (int jj = 0; jj < bN; jj++)
						tmp += acc_img[(stride_i + ii) * width + stride_j + jj];
				}
				float mean = tmp / (bM * bN);

				if (bit)
					acc_img[(stride_i + i_insert) * width + stride_j + j_insert] = fabsf(mean); //+
				else
					acc_img[(stride_i + i_insert) * width + stride_j + j_insert] = -1.0f * fabsf(mean); //-

				bj++;
				if (bj >= bsJ)
				{
					bj = 0;
					bi++;
				}
			});
		}).wait();	
	};
}

void extract_msg(
				sycl::buffer<float, 1>& img,
				int width, int height, 
				sycl::buffer<char, 1>& msg, 
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
			
			sycl::accessor acc_img{img, h, sycl::read_write};
			sycl::accessor acc_msg{msg, h, sycl::read_write};
			h.parallel_for(sycl::range<1>(msg_length), [=](sycl::id<1> item){
				char ch = 0;
				auto c = item;

				int bi = 0;
				int bj = 0;

				for (int b = 0; b < 8; b++)
				{
					int bit;

					int stride_i = bi * bM;
					int stride_j = bj * bN;
					float tmp = 0.0;
					for (int ii = 0; ii < bM; ii++)
					{
						for (int jj = 0; jj < bN; jj++)
							tmp += acc_img[(stride_i + ii) * width + stride_j + jj];
					}
					float mean = tmp / (bM * bN);

					if (acc_img[(stride_i + i_insert) * width + stride_j + j_insert] > 0.5f * mean)
						bit = 1;
					else
						bit = 0;

					ch = (bit << b) | ch;

					bj++;
					if (bj >= bsJ)
					{
						bj = 0;
						bi++;
					}
				}
				acc_msg[c] = ch;
			});
		}).wait();	
	};
}

void encoder(char *file_in, char *file_out, char *c_msg, int msg_len, sycl::queue &Q)
{

	int w, h;
	// im corresponds to R0G0B0,R1G1B1.....
	uint8_t *im = (uint8_t *)loadPNG(file_in, &w, &h);
	uint8_t *im_out = (uint8_t *)malloc(3 * w * h * sizeof(uint8_t));
	{
		// auto size = sycl::range<1>(w * h);
		// sycl::buffer<uint8_t, 1> imgIn(im, size);
		// sycl::buffer<float, 1> imgX(size);
		// sycl::buffer<float, 1> imgY(size);
		// sycl::buffer<float, 1> imgZ(size);
		// sycl::buffer<uint8_t, 1> imgOut(im_out, size);
		// sycl::buffer<float, 1> Ydct(sycl::range<1>(w * h));
		// sycl::buffer<float, 1> mcosine(sycl::range<1>(8 * 8));
		// sycl::buffer<float, 1> alpha(sycl::range<1>(8));
		// sycl::buffer<char, 1> msg(c_msg, sycl::range<1>(msg_len));

		uint8_t* _imgIn = sycl::malloc_device<uint8_t>(w * h, Q);//memset
		float* _imgX = sycl::malloc_device<float>(w * h, Q);
		float* _imgY = sycl::malloc_device<float>(w * h, Q);
		float* _imgZ = sycl::malloc_device<float>(w * h, Q);
		uint8_t* _imgOut = sycl::malloc_device<uint8_t>(w * h, Q);
		float* _Ydct = sycl::malloc_device<float>(w * h, Q);
		float* _mcosine = sycl::malloc_device<float>(8 * 8, Q);
		float* _alpha = sycl::malloc_device<float>(8, Q);
		char* _msg = sycl::malloc_device<char>(msg_len, Q);

		Q.memcpy(_imgIn, im, sizeof(uint8_t) * (w * h));

		
		// Create imRGB & imYCrCb
		get_dct8x8_params(_mcosine, _alpha, Q);
		double start = omp_get_wtime();

		im2ycbcr(_imgIn, _imgX, _imgY, _imgZ, w, h, Q);
		dct8x8_2d(_imgX, _Ydct, w, h, _mcosine, _alpha, Q);

		// Insert Message
		insert_msg(_Ydct, w, h, _msg, msg_len, Q);

		idct8x8_2d(_imgX, _Ydct, w, h, _mcosine, _alpha, Q);
 
		ycbcr2im(_imgOut, _imgX, _imgY, _imgZ, w, h, Q);

		double stop = omp_get_wtime();
		printf("Encoding time=%f sec.\n", stop - start);

		Q.memcpy(im_out, _imgOut, sizeof(uint8_t) * (w * h));

		savePNG(file_out, (uint8_t*)im_out, w, h);

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
		// auto size = sycl::range<1>(w * h);
		// sycl::buffer<uint8_t, 1> imgIn(im, size);
		// sycl::buffer<float, 1> imgX(size);
		// sycl::buffer<float, 1> imgY(size);
		// sycl::buffer<float, 1> imgZ(size);
		// sycl::buffer<float, 1> Ydct(sycl::range<1>(w * h));
		// sycl::buffer<float, 1> mcosine(sycl::range<1>(8 * 8));
		// sycl::buffer<float, 1> alpha(sycl::range<1>(8));
		// sycl::buffer<char, 1> msg(msg_decoded, sycl::range<1>(msg_len));
		
		uint8_t* _imgIn = sycl::malloc_device<uint8_t>(w * h, Q);//memset
		float* _imgX = sycl::malloc_device<float>(w * h, Q);
		float* _imgY = sycl::malloc_device<float>(w * h, Q);
		float* _imgZ = sycl::malloc_device<float>(w * h, Q);
		float* _Ydct = sycl::malloc_device<float>(w * h, Q);
		float* _mcosine = sycl::malloc_device<float>(8 * 8, Q);
		float* _alpha = sycl::malloc_device<float>(8, Q);
		char* _msg = sycl::malloc_device<char>(msg_len, Q);

		Q.memcpy(_imgIn, im, sizeof(uint8_t) * (w * h));

		// Create imRGB & imYCrCb

		get_dct8x8_params(_mcosine, _alpha, Q);

		double start = omp_get_wtime();

		// im2imRGB(im, w, h, &imRGB);
		// rgb2ycbcr(&imRGB, &imYCrCb, Q);
		im2ycbcr(_imgIn, _imgX, _imgY, _imgZ, w, h, Q);
		dct8x8_2d(_imgX, _Ydct, w, h, _mcosine, _alpha, Q);

		extract_msg(_Ydct, w, h, _msg, msg_len, Q);

		double stop = omp_get_wtime();

		Q.memcpy(msg_decoded, _msg, sizeof(char) * (msg_len));

		printf("Decoding time=%f sec.\n", stop - start);

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
