// Allocate Memory: Width*Height*sizeof(int)
// Use GPU to calculate iter for each pixel
// Pass that back to the host who then calculates the colour and creates the bitmap

/*
MEMORY
xoffset, yoffset, resolution shared?

*/

extern "C" {
    #include "bmpfile.h"
}

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

/*Colour Values*/
#define COLOUR_DEPTH 255
#define COLOUR_MAX 240.0
#define GRADIENT_COLOUR_MAX 230.0

#define FILENAME "my_mandelbrot_fractal.bmp"

 /** 
   * Computes the color gradiant
   * color: the output vector 
   * x: the gradiant (beetween 0 and 360)
   * min and max: variation of the RGB channels (Move3D 0 -> 1)
   * Check wiki for more details on the colour science: en.wikipedia.org/wiki/HSL_and_HSV 
   */
void GroundColorMix(double* color, double x, double min, double max)
{
  /*
   * Red = 0
   * Green = 1
   * Blue = 2
   */
    double posSlope = (max-min)/60;
    double negSlope = (min-max)/60;

    if( x < 60 )
    {
        color[0] = max;
        color[1] = posSlope*x+min;
        color[2] = min;
        return;
    }
    else if ( x < 120 )
    {
        color[0] = negSlope*x+2.0*max+min;
        color[1] = max;
        color[2] = min;
        return;
    }
    else if ( x < 180  )
    {
        color[0] = min;
        color[1] = max;
        color[2] = posSlope*x-2.0*max+min;
        return;
    }
    else if ( x < 240  )
    {
        color[0] = min;
        color[1] = negSlope*x+4.0*max+min;
        color[2] = max;
        return;
    }
    else if ( x < 300  )
    {
        color[0] = posSlope*x-4.0*max+min;
        color[1] = min;
        color[2] = max;
        return;
    }
    else
    {
        color[0] = max;
        color[1] = min;
        color[2] = negSlope*x+6*max;
        return;
    }
}

int parse_args(int argc, char *argv[], int *height, int *width)
{
    if ( (argc != 3) ||
        (*height = atoi(argv[1])) <= 0 ||
        (*width = atoi(argv[2])) <= 0) {
            fprintf(stderr, "Usage: ./%s height width\n", argv[0]);
            return -1;
        }
    return 0;
}

__global__ void calcMandelbrot(int* out, int height, int width, float resolution)
{
    // Get individual threadId
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Setup local variables
    int totalPixels = height*width;
    double xoffset = -(width - 1)/2.0;
    double yoffset = (height - 1)/2.0;
    double xcenter = -0.55;
    double ycenter = 0.6;
    int max_iter = 1000;

    // For any thread within the image
    while (id < totalPixels) {
        
        int col = id % width;
        int row = id / width;
        
		
        int currentIndex = col + (row * width);
        
        double x = xcenter + (xoffset + col)/resolution;
        double y = ycenter + (yoffset - row)/resolution;
        
        int iter = 0;
        double a = 0.0, b = 0.0, a_old = 0.0, b_old = 0.0;
        double mag_sqr = 0.0;
        
        while (iter < max_iter && mag_sqr <= 4.0)
		{
			iter++;
			a = a_old*a_old - b_old*b_old + x;
			b = 2.0*a_old*b_old + y;
			mag_sqr = a*a + b*b;
			a_old = a;
            b_old = b;
        }
        out[currentIndex] = iter;
        id += blockDim.x * gridDim.x;
    }
}

int main(int argc, char* argv[])
{
    // Setup BMP file
    bmpfile_t *bmp;
    rgb_pixel_t pixel = {0, 0, 0, 0};

    int height, width;
    if (parse_args(argc, argv, &height, &width) < 0) exit(EXIT_FAILURE);
    
    // Setup Memory
    int image_size = width*height;
    int *dev_image;
    float resolution = image_size/240;
    bmp = bmp_create(width, height, 32);
    // Allocate memory for image as vector
    cudaMalloc((void**)&dev_image, image_size*sizeof(int));
    int *host_image = (int *)malloc(image_size*sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (image_size + threadsPerBlock - 1)/ threadsPerBlock;

    calcMandelbrot<<<blocksPerGrid, threadsPerBlock>>>(dev_image, height, width, resolution);
    // Copy iteration array back to host
    cudaMemcpy(host_image, dev_image, (height*width*sizeof(int)), cudaMemcpyDeviceToHost);
    
    double x_col;
    double color[3];
    int col = 0;
    int row = 0;
    for (int i = 0; i < image_size; i++) {        
        x_col = (240.0 - (( (((float)host_image[i] / ((float) 1000)) * 230.0))));
        GroundColorMix(color, x_col, 1, 255);
        pixel.red = color[0];
        pixel.green = color[1];
	    pixel.blue = color[2];
        bmp_set_pixel(bmp, col, row, pixel);
        // Simulate the effect of iterating through a 2D image
        if (col == width - 1) {
            col = -1;
            row++;
        }
        col++;
    }

    bmp_save(bmp, FILENAME);
    bmp_destroy(bmp);
    cudaFree(dev_image);
    free(host_image);
    
    return 0;
}