COMPILER = nvcc
CFLAGS = -I /usr/local/cuda-9.2/samples/common/inc 
EXES = parallel_mandelbrot
all = ${EXES}

parallel_mandelbrot: parallel_mandelbrot.cu bmpfile.o
	${COMPILER} ${CFLAGS} -o parallel_mandelbrot parallel_mandelbrot.cu bmpfile.o

bmpfile.o: bmpfile.c bmpfile.h
	${COMPILER} -c bmpfile.c

clean:
	rm -f *.o *~ ${EXES}