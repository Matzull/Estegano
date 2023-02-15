#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>
#include <sycl/sycl.hpp>

#include "io_routines.h"
#include "stegano_routines.h"

#define M_PI 3.14159265358979323846 /* pi */

void im2imRGB(uint8_t *im, int w, int h, t_sRGB *imRGB)
{
	imRGB->w = w;
	imRGB->h = h;

	float *R = imRGB->R;
	float *G = imRGB->G;
	float *B = imRGB->B;

	int index = 0;
	int three_index = 0;

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			R[index] = im[three_index];
			G[index] = im[three_index + 1];
			B[index] = im[three_index + 2];
			three_index += 3;
			index += 1;
		}
	}
}

// void im2imRGB(uint8_t *im, int w, int h, t_sRGB *imRGB, sycl::queue Q)
// {
// 	imRGB->w = w;
// 	imRGB->h = h;
// 	{
// 		sycl::buffer im_buff{im};
// 		Q.submit([&](sycl::handler &h){
// 			sycl::accessor acc_im{im_buff, h, sycl::read_write};
// 			h.parallel_for(range<2>(w, h), [=](id<2> item){
// 				imRGB->R[i * w + j] = im[3 * (i * w + j)];
// 				imRGB->G[i * w + j] = im[3 * (i * w + j) + 1];
// 				imRGB->B[i * w + j] = im[3 * (i * w + j) + 2];
// 			});
// 		}).wait();
// 	};
// }

void imRGB2im(t_sRGB *imRGB, uint8_t *im, int &w, int &h)
{
	w = imRGB->w;
	h = imRGB->h;

	float *R = imRGB->R;
	float *G = imRGB->G;
	float *B = imRGB->B;

	int index = 0;
	int three_index = 0;

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			im[three_index] = R[index];
			im[three_index + 1] = G[index];
			im[three_index + 2] = B[index];
			three_index += 3;
			index += 1;
		}
	}
}

// funtion for translate from RGB to YCbCr
void rgb2ycbcr(t_sRGB *in, t_sYCrCb *out, sycl::queue& Q)
{
	out->w = in->w;
	out->h = in->h;

	auto size = sycl::range<1>(out->w * out->h);

	int index = 0;
	{
		sycl::buffer<float, 1> in_buff_R(in->R, size);
		sycl::buffer<float, 1> in_buff_G(in->G, size);
		sycl::buffer<float, 1> in_buff_B(in->B, size);
		sycl::buffer<float, 1> out_buff_Y(out->Y, size);
		sycl::buffer<float, 1> out_buff_Cb(out->Cb, size);
		sycl::buffer<float, 1> out_buff_Cr(out->Cr, size);

		Q.submit([&](sycl::handler &h){
			sycl::accessor acc_in_R{in_buff_R, h, sycl::read_only};
			sycl::accessor acc_in_G{in_buff_G, h, sycl::read_only};
			sycl::accessor acc_in_B{in_buff_B, h, sycl::read_only};

			sycl::accessor acc_out_Y{out_buff_Y, h, sycl::write_only};
			sycl::accessor acc_out_Cb{out_buff_Cb, h, sycl::write_only};
			sycl::accessor acc_out_Cr{out_buff_Cr, h, sycl::write_only};

			h.parallel_for(sycl::range<1>(out->w * out->h), [=](sycl::id<1> item){
				acc_out_Y[item] = 0.299 * acc_in_R[item] + 0.587 * acc_in_G[item] + 0.114 * acc_in_B[item];
				acc_out_Cb[item] = 128.0 + 0.5 * acc_in_R[item] - 0.418688 * acc_in_G[item] - 0.081312 * acc_in_B[item];
				acc_out_Cr[item] = 128.0 - 0.168736 * acc_in_R[item] - 0.3331264 * acc_in_G[item] + 0.5 * acc_in_B[item];
			});
		}).wait();
	};
}

// function for translate YCbCr to RGB
void ycbcr2rgb(t_sYCrCb *in, t_sRGB *out)
{

	int w = in->w;
	out->w = in->w;
	out->h = in->h;

	for (int i = 0; i < in->h; i++)
	{
		for (int j = 0; j < in->w; j++)
		{

			// Use standard coeficient
			out->R[i * w + j] = in->Y[i * w + j] + 1.402 * (in->Cb[i * w + j] - 128.0);
			out->G[i * w + j] = in->Y[i * w + j] - 0.34414 * (in->Cr[i * w + j] - 128.0) - 0.71414 * (in->Cb[i * w + j] - 128.0);
			out->B[i * w + j] = in->Y[i * w + j] + 1.772 * (in->Cr[i * w + j] - 128.0);

			// After translate we must check if RGB component is in [0...255]
			if (out->R[i * w + j] < 0)
				out->R[i * w + j] = 0;
			else if (out->R[i * w + j] > 255)
				out->R[i * w + j] = 255;

			if (out->G[i * w + j] < 0)
				out->G[i * w + j] = 0;
			else if (out->G[i * w + j] > 255)
				out->G[i * w + j] = 255;

			if (out->B[i * w + j] < 0)
				out->B[i * w + j] = 0;
			else if (out->B[i * w + j] > 255)
				out->B[i * w + j] = 255;
		}
	}
}

void im2ycbcr(
				sycl::buffer<float, 1>& imgIn, 
				sycl::buffer<float, 1>& imgX, 
				sycl::buffer<float, 1>& imgY, 
				sycl::buffer<float, 1>& imgZ,
				int w, int h,
				sycl::queue& Q
			)
{
	{
		Q.submit([&](sycl::handler &h){
			sycl::accessor acc_imgIn{in_buff_R, h, sycl::read_only};
			sycl::accessor acc_out_Y{imgX, h, sycl::write_only};
			sycl::accessor acc_out_Cb{imgY, h, sycl::write_only};
			sycl::accessor acc_out_Cr{imgZ, h, sycl::write_only};

			h.parallel_for(sycl::range<1>(h * w), [=](sycl::id<1> item){
				auto t_i = item + item + item;
				acc_out_Y[item] = 0.299 * acc_imgIn[t_i] + 0.587 * acc_imgIn[t_i + 1] + 0.114 * acc_imgIn[t_i + 2];
				acc_out_Cb[item] = 128.0 + 0.5 * acc_imgIn[t_i] - 0.418688 * acc_imgIn[t_i + 1] - 0.081312 * acc_imgIn[t_i + 2];
				acc_out_Cr[item] = 128.0 - 0.168736 * acc_imgIn[t_i] - 0.3331264 * acc_imgIn[t_i + 1] + 0.5 * acc_imgIn[t_i + 2];
			});
		}).wait();
	};	
}

void ycbcr2im(	
				sycl::buffer<float, 1>& imgOut, 
				sycl::buffer<float, 1>& imgX, 
				sycl::buffer<float, 1>& imgY, 
				sycl::buffer<float, 1>& imgZ,
				int w, int h,
				sycl::queue& Q
			)
{
	{
		Q.submit([&](sycl::handler &h){
			sycl::accessor acc_imgOut{in_buff_R, h, sycl::read_only};
			sycl::accessor acc_out_Y{imgX, h, sycl::write_only};
			sycl::accessor acc_out_Cb{imgY, h, sycl::write_only};
			sycl::accessor acc_out_Cr{imgZ, h, sycl::write_only};

			h.parallel_for(sycl::range<1>(h * w), [=](sycl::id<1> item){
				auto t_i = item + item + item;
				
				acc_imgOut[t_i] 	= sycl::clamp(acc_out_Y[item] + 1.402 * (acc_out_Cb[item] - 128.0), 0, 255);
				acc_imgOut[t_i + 1] = sycl::clamp(acc_out_Y[item] - 0.34414 * (acc_out_Cr[item] - 128.0) - 0.71414 * (acc_out_Cb[item] - 128.0), 0, 255);
				acc_imgOut[t_i + 2] = sycl::clamp(acc_out_Y[item] + 1.772 * (acc_out_Cr[item] - 128.0), 0, 255);
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

			h.single_task([=](sycl::id<1> item){
				for (int i = 0; i < bM; i++)
				for (int j = 0; j < bN; j++)
				{
					acc_mcosine[i * bN + j] = cos(((2 * i + 1) * M_PI * j) / (2 * bM));
				}
				alpha[0] = 1 / sqrt(bM * 1.0f);
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

			h.parallel_for(sycl::range<2>(width / bN, heigth / bM), [=](sycl::id<2> item){
				auto stride_i = item[0] * bM;
				auto stride_j = item[1] * bN;
				for (int i = 0; i < bM; i++)
				{
					for (int j = 0; j < bN; j++)
					{
						float tmp = 0.0;
						for (int ii = 0; ii < bM; ii++)
						{
							for (int jj = 0; jj < bN; jj++)
								tmp += acc_in[(stride_i + ii) * width + stride_j + jj] * acc_mcosine[ii * bN + i] * acc_mcosine[jj * bN + j];
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

			h.parallel_for(sycl::range<2>(width / bN, height / bM), [=](sycl::id<2> item){
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

void insert_msg(float *img, int width, int height, char *msg, int msg_length)
{
	int i_insert = 3;
	int j_insert = 4;

	int bM = 8;
	int bN = 8;

	int bsI = height / bM;
	int bsJ = width / bN;
	int bi = 0;
	int bj = 0;

	if (bsI * bsJ < msg_length * 8)
		printf("Image not enough to save message!!!\n");

	for (int c = 0; c < msg_length; c++)
	{
		for (int b = 0; b < 8; b++)
		{
			char ch = msg[c];
			char bit = (ch & (1 << b)) >> b;

			int stride_i = bi * bM;
			int stride_j = bj * bN;
			float tmp = 0.0;
			for (int ii = 0; ii < bM; ii++)
			{
				for (int jj = 0; jj < bN; jj++)
					tmp += img[(stride_i + ii) * width + stride_j + jj];
			}
			float mean = tmp / (bM * bN);

			if (bit)
				img[(stride_i + i_insert) * width + stride_j + j_insert] = fabsf(mean); //+
			else
				img[(stride_i + i_insert) * width + stride_j + j_insert] = -1.0f * fabsf(mean); //-

			bj++;
			if (bj >= bsJ)
			{
				bj = 0;
				bi++;
			}
		}		
	}
		
}

void extract_msg(float *img, int width, int height, char *msg, int msg_length)
{
	int i_insert = 3;
	int j_insert = 4;

	int bM = 8;
	int bN = 8;

	int bsI = height / bM;
	int bsJ = width / bN;
	int bi = 0;
	int bj = 0;

	for (int c = 0; c < msg_length; c++)
	{
		char ch = 0;

		for (int b = 0; b < 8; b++)
		{
			int bit;

			int stride_i = bi * bM;
			int stride_j = bj * bN;
			float tmp = 0.0;
			for (int ii = 0; ii < bM; ii++)
			{
				for (int jj = 0; jj < bN; jj++)
					tmp += img[(stride_i + ii) * width + stride_j + jj];
			}
			float mean = tmp / (bM * bN);

			if (img[(stride_i + i_insert) * width + stride_j + j_insert] > 0.5f * mean)
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
		msg[c] = ch;
	}
}

void encoder(char *file_in, char *file_out, char *msg, int msg_len, sycl::queue &Q)
{

	int w, h, bytesPerPixel;

	// im corresponds to R0G0B0,R1G1B1.....
	uint8_t *im = (uint8_t *)loadPNG(file_in, &w, &h);
	uint8_t *im_out = (uint8_t *)malloc(3 * w * h * sizeof(uint8_t));
	{
		auto size = sycl::range<1>(w * h);
		sycl::buffer<float, 1> imgIn(im, size);
		sycl::buffer<float, 1> imgX(size);
		sycl::buffer<float, 1> imgY(size);
		sycl::buffer<float, 1> imgZ(size);
		sycl::buffer<float, 1> imgOut(size);
		sycl::buffer<float, 1> Ydct(sycl::range<1>(w * h));
		sycl::buffer<float, 1> mcosine(sycl::range<1>(8 * 8));
		sycl::buffer<float, 1> alpha(sycl::range<1>(8));
		
		// Create imRGB & imYCrCb
		
		get_dct8x8_params(mcosine, alpha, Q);

		double start = omp_get_wtime();

		// im2imRGB(im, w, h, &imRGB);
		// rgb2ycbcr(&imRGB, &imYCrCb, Q);
		im2ycbcr(imgIn, imgX, imgY, imgZ, w, h, Q);
		dct8x8_2d(imgX, Ydct, w, h, mcosine, alpha, Q);

		// Insert Message
		insert_msg(Ydct, imYCrCb.w, imYCrCb.h, msg, msg_len);

		idct8x8_2d(imgX, Ydct, w, h, mcosine, alpha, Q);
		// ycbcr2rgb(&imYCrCb, &imRGB);
		// imRGB2im(&imRGB, im_out, w, h);
		ycbcr2im(imgOut, imgX, imgY, imgZ, w, h, Q);

		double stop = omp_get_wtime();
		printf("Encoding time=%f sec.\n", stop - start);

		savePNG(file_out, im_out, w, h);
	};
}

void decoder(char *file_in, char *msg_decoded, int msg_len, sycl::queue &Q)
{

	int w, h, bytesPerPixel;

	uint8_t *im = (uint8_t *)loadPNG(file_in, &w, &h);
	uint8_t *im_out = (uint8_t *)malloc(3 * w * h * sizeof(uint8_t));

	{
		auto size = sycl::range<1>(w * h);
		sycl::buffer<float, 1> imgIn(im, size);
		sycl::buffer<float, 1> imgX(size);
		sycl::buffer<float, 1> imgY(size);
		sycl::buffer<float, 1> imgZ(size);
		sycl::buffer<float, 1> imgOut(size);
		sycl::buffer<float, 1> Ydct(sycl::range<1>(w * h));
		sycl::buffer<float, 1> mcosine(sycl::range<1>(8 * 8));
		sycl::buffer<float, 1> alpha(sycl::range<1>(8));
		
		// Create imRGB & imYCrCb

		get_dct8x8_params(mcosine, alpha, Q);

		double start = omp_get_wtime();

		// im2imRGB(im, w, h, &imRGB);
		// rgb2ycbcr(&imRGB, &imYCrCb, Q);
		im2ycbcr(imgIn, imgX, imgY, imgZ, w, h, Q);
		dct8x8_2d(imgX, Ydct, w, h, mcosine, alpha, Q);

		extract_msg(Ydct, imYCrCb.w, imYCrCb.h, msg_decoded, msg_len);

		double stop = omp_get_wtime();
		printf("Decoding time=%f sec.\n", stop - start);
	};
}
