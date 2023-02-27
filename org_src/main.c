#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "stegano_routines.h"
#include "io_routines.h"

int main(int argc, char **argv)
{
	char file_in[20], file_out[20], file_logo[20];

	if (argc<1){
		printf("./exec image_in.png logo.png image_out.png\n");
		return(-1);
	}

	strcpy_s(file_in, strlen(argv[1]) + 1, argv[1]);
	strcpy_s(file_logo, strlen(argv[2]) + 1, argv[2]);
	strcpy_s(file_out, strlen(argv[3]) + 1, argv[3]);

	char *msg, *msg_decoded;
	int msg_len;
	get_msg(".\\imgs\\logo_out.png", &msg, &msg_len);

	encoder(".\\imgs\\image_skyscraper.png", file_out, msg, msg_len);

	// Extract msg from image
	msg_decoded = (char*)malloc(msg_len);
	decoder(file_out, msg_decoded, msg_len);
	printf("Message len: %d", msg_len);
	msg2logo(".\\imgs\\Out\\logo_out.png", msg_decoded, msg_len);

	return(0);
}
