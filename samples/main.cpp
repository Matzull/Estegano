#include <iostream>

void ycbcr2rgb(float* imY, float* imCb, float* imCr, int w, int h){

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {

			// Use standard coeficient
			out->R[i*w+j] = in->Y[i*w+j]                                 + 1.402*(in->Cb[i*w+j]-128.0);
			out->G[i*w+j] = in->Y[i*w+j] - 0.34414*(in->Cr[i*w+j]-128.0) - 0.71414*(in->Cb[i*w+j]-128.0); 
			out->B[i*w+j] = in->Y[i*w+j] + 1.772*(in->Cr[i*w+j]-128.0);
			
			// After translate we must check if RGB component is in [0...255]
			if (out->R[i*w+j] < 0) out->R[i*w+j] = 0;
			else if (out->R[i*w+j] > 255) out->R[i*w+j] = 255;

			if (out->G[i*w+j] < 0) out->G[i*w+j] = 0;
			else if (out->G[i*w+j] > 255) out->G[i*w+j] = 255;

			if (out->B[i*w+j] < 0) out->B[i*w+j]= 0;
			else if (out->B[i*w+j] > 255) out->B[i*w+j] = 255;
		}
	}
}


void imRGB2im(t_sRGB *imRGB, uint8_t *im, int *w, int *h)
{
	int w_ = imRGB->w;
	*w = imRGB->w;
	*h = imRGB->h;

	for (int i=0; i<*h; i++)
		for (int j=0; j<*w; j++)
		{
			im[3*(i*w_+j)  ] = imRGB->R[i*w_+j];
			im[3*(i*w_+j)+1] = imRGB->G[i*w_+j];  
			im[3*(i*w_+j)+2] = imRGB->B[i*w_+j];    
		}                    
}

int main(int argc, char **argv) {




  return 0;
}