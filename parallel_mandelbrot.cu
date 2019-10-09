extern "C" {
    #include "bmpfile.h"
}

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

/*
 * Parallel Mandelbrot
 * --------------------
 * Author: Matthew Gray
 * Student No: 220186070
 * Email: mgray44@myune.edu.au
 *
 * This program uses the CUDA GPU to calculate
 * Mandelbrot fractals, using parallelished
 * threads.
 *
 * To compile:
 * make parallel_mandelbrot or make
 *
 * To run:
 * parallel_mandelbrot imageHeight imageWidth
 *
 */

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

/*
 * Function that parses command line arguments
 * 
 * Parameters
 *------------
 * argc: Argument count
 * argv[]: array containing arguments
 * *height: pointer to location to store image height
 * *width: location to store image width
 * 
 * Returns
 * --------
 * 0 if arguments correct, -1 otherwise
 *
 */
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

/*
 * Function that calculcates the iterations for the mandelbrot
 * algorithm for each pixel in the image and stores it in an
 * array.
 *
 * Parameters
 * -----------
 * *out: Array containing iterations for each pixel
 * height: Image height
 * width: Image width
 * resolution: Image resolution
 *
 */
__global__ void calcMandelbrot(int* out, int height, int width, float resolution)
{
    // Get individual threadId
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // Setup shared block data
    __shared__ double data[4];

    // First thread in each block initialises the data
    if (threadIdx.x == 0) {
        data[0] = -(width - 1)/2.0; // xoffset
        data[1] = (height - 1)/2.0; // yoffset
        data[2] = -0.55;    // xcenter
        data[3] = 0.6;      // ycenter
    }

    __syncthreads();

    int totalPixels = height*width;
    int max_iter = 1000;

    // For any thread within the image
    while (id < totalPixels) {
        
        // Calculate thread pixel location within image
        int col = id % width;
        int row = id / width;
        
		// Use that to calculate its index within the image array
        int currentIndex = col + (row * width);
        
        double x = data[2] + (data[0] + col)/resolution;
        double y = data[3] + (data[1] - row)/resolution;
        
        int iter = 0;
        double a = 0.0, b = 0.0, a_old = 0.0, b_old = 0.0;
        double mag_sqr = 0.0;
        
        // Use the algorithm for determining if it's contained within the set
        while (iter < max_iter && mag_sqr <= 4.0)
		{
			iter++;
			a = a_old*a_old - b_old*b_old + x;
			b = 2.0*a_old*b_old + y;
			mag_sqr = a*a + b*b;
			a_old = a;
            b_old = b;
        }
        // Store iteration in image array
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
    // Calculate image resolution based of input image size
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
    // Iterate through image vector
    for (int i = 0; i < image_size; i++) {
        // Calculate pixel colour based off of iterations        
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
    // Free memory
    cudaFree(dev_image);
    free(host_image);
    
    return 0;
}