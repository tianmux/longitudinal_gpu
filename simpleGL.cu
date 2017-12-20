////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/*
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include "helper_gl.h"
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     5 //ms

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 1024;
const unsigned int window_height = 1024;


const unsigned int Nring = 1<<6;
const unsigned int n_per_ring = 1<<10;
const unsigned int Npar = n_per_ring*Nring;
const unsigned int n_per_show = 10;
double Qx = 3.666;
double nux = 2.0*3.1415926*Qx;
double sinnux = sin(nux);
double cosnux = cos(nux);
double Qy = 3.7;
double nuy = 2.0*3.1415926*Qy;
double sinnuy = sin(nuy);
double cosnuy = cos(nuy);
double S = 0.0;
double bt = 1.0;
double v1 = 0;//used to manually control the voltage.
double v2 = 1;

unsigned int initial = 0;

// vbo variables
GLuint vbo;
GLuint vbox;
struct cudaGraphicsResource *cuda_vbo_resource;
struct cudaGraphicsResource *cuda_vbox_resource;
void *d_vbo_buffer = NULL;
void *d_vbox_buffer = NULL;
float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -1;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
float GFLOPS = 0;        // GFLOPS
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);
void runAutoTest(int devID, char **argv, char *ref_file);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

const char *sSDKsample = "simpleGL (VBO)";

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////

__global__ void init_kernel(double4 *pos, unsigned int Npar)
{
    unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

    double xx = 0.5*(double)(blockIdx.x+0.0)/(double)Nring*cosf(2*3.1415926*(double)threadIdx.x/(double)blockDim.x);
    double yy = 0.5*(double)(blockIdx.x+0.0)/(double)Nring*cosf(2*3.1415926*(double)threadIdx.x/(double)blockDim.x);
    double px = 0.5*(double)(blockIdx.x+0.0)/(double)Nring*sinf(2*3.1415926*(double)threadIdx.x/(double)blockDim.x);
    double py = 0.5*(double)(blockIdx.x+0.0)/(double)Nring*sinf(2*3.1415926*(double)threadIdx.x/(double)blockDim.x);

    // write output vertex
    pos[n] = make_double4(xx,px,yy,py);
}

__global__ void simple_vbo_kernel(double4 *pos, double v1, double v2)
{
    unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;
    double x = pos[n].x;
    double px = pos[n].y;

    for(unsigned int i = 0;i<n_per_show;++i){
        
        px = px+1e-3*(1-v1)*(sinf(x))+1e-3*(1-v2)*(sinf(2.0*x)-1);
        x = x-0.001*px;
    }
/*
    x = cosnux*xx+bt*sinnux*pxx;
    y = cosnuy*yy+bt*sinnuy*pyy;
    px = -1.0/bt*sinnux*xx+cosnux*pxx-0.5*S*(x*x-y*y);
    py = -1.0/bt*sinnuy*yy+cosnuy*pyy-S*x*y;
*/
    //if (x > 5 || y >5 || px>5||py>5){
    //    x = 0;
    //    y = 0;
    //    px = 0;
    //   py = 0;
    //}
    // write output vertex
    pos[n] = make_double4(x,px,0,0);
}


void launch_kernel(double4 *pos,  unsigned int Npar,double nu,double S,double bt)
{
    // execute the kernel
    dim3 block(1024, 1, 1);
    dim3 grid(Npar/ block.x>1?Npar/ block.x:1, 1, 1);
    simple_vbo_kernel<<< grid, block>>>(pos,v1,v2);
}

bool checkHW(char *name, const char *gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

int findGraphicsGPU(char *name)
{
    int nGraphicsGPU = 0;
    int deviceCount = 0;
    bool bFoundGraphics = false;
    char firstGraphicsName[256], temp[256];

    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("> FAILED %s sample finished, exiting...\n", sSDKsample);
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("> There are no device(s) supporting CUDA\n");
        return false;
    }
    else
    {
        printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
        printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

        if (bGraphics)
        {
            if (!bFoundGraphics)
            {
                strcpy(firstGraphicsName, temp);
            }

            nGraphicsGPU++;
        }
    }

    if (nGraphicsGPU)
    {
        strcpy(name, firstGraphicsName);
    }
    else
    {
        strcpy(name, "this hardware");
    }

    return nGraphicsGPU;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    char *ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("%s starting...\n", sSDKsample);

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            // In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
            getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
        }
    }

    printf("\n");

    runTest(argc, argv, ref_file);

    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);
        GFLOPS = 18.0f*Npar*n_per_show/(sdkGetAverageTimerValue(&timer) / 1000.f)/1000000000;
        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "#Particles:%4d; #Turns: %4d; GFLOPS: %4.4f, Time per turn (us): %4.4f", Npar,frameCount*n_per_show,GFLOPS,sdkGetAverageTimerValue(&timer)/n_per_show*1000);
    glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();

    return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // command line mode only
    if (ref_file != NULL)
    {
        // This will pick the best possible CUDA capable device
        int devID = findCudaDevice(argc, (const char **)argv);

        // create VBO
        checkCudaErrors(cudaMalloc((void **)&d_vbo_buffer, Npar*4*sizeof(double)));

        // run the cuda part
        runAutoTest(devID, argv, ref_file);

        // check result of Cuda step
        checkResultCuda(argc, argv, vbo);

        cudaFree(d_vbo_buffer);
        d_vbo_buffer = NULL;
    }
    else
    {
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        if (false == initGL(&argc, argv))
        {
            return false;
        }

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        if (checkCmdLineFlag(argc, (const char **)argv, "device"))
        {
            if (gpuGLDeviceInit(argc, (const char **)argv) == -1)
            {
                return false;
            }
        }
        else
        {
            cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
        }

        // register callbacks
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif

        // create VBO
        createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

        // run the cuda part
        runCuda(&cuda_vbo_resource);

        // start rendering mainloop
        glutMainLoop();
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    double4 *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    // execute the kernel
    //    dim3 block(8, 8, 1);
    //    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    //    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);
    if (initial == 0){
        init_kernel<<<Npar/n_per_ring,n_per_ring>>>(dptr,Npar);
        initial =1;
    }
    launch_kernel(dptr, Npar, nux, S, bt);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif

void sdkDumpBin2(void *data, unsigned int bytes, const char *filename)
{
    printf("sdkDumpBin: <%s>\n", filename);
    FILE *fp;
    FOPEN(fp, filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int devID, char **argv, char *ref_file)
{
    char *reference_file = NULL;
    void *imageData = malloc(Npar*sizeof(double));

    // execute the kernel
    launch_kernel((double4 *)d_vbo_buffer, Npar, nux,S,bt);

    cudaDeviceSynchronize();
    getLastCudaError("launch_kernel failed");

    checkCudaErrors(cudaMemcpy(imageData, d_vbo_buffer, Npar*sizeof(double), cudaMemcpyDeviceToHost));

    sdkDumpBin2(imageData, Npar*sizeof(double), "simpleGL.bin");
    reference_file = sdkFindFilePath(ref_file, argv[0]);

    if (reference_file &&
        !sdkCompareBin2BinFloat("simpleGL.bin", reference_file,
                                Npar*sizeof(double),
                                MAX_EPSILON_ERROR, THRESHOLD, pArgv[0]))
    {
        g_TotalErrors++;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = Npar * 4 * sizeof(double);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    bt-=0.0001;
    runCuda(&cuda_vbo_resource);
    g_fAnim += 0.01f;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
    glPointSize(4.0);
    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(2, GL_DOUBLE, 4*sizeof(double), 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, Npar);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    sdkStopTimer(&timer);
    computeFPS();

}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
        case ('r'):
            initial = 0;
            break;
        case ('a'): // left arrow key reduce sextuple strength
            S-=0.1;
            break;
        case ('d'):// right arrow key reduce sextuple strength
            S+=0.1;
            break;
        case ('w'):// up arrow key increase tune
            Qx+=0.01;
            nux = 2.0*3.1415926*Qx;
            break;
        case ('s'):// down arrow key decrease tune
            Qx-=0.01;
            nux = 2.0*3.1415926*Qx;
            break;
        case ('f'):// f key increase second order harmonic
            v2 +=0.01;
            break; 
        case ('v'):// g key decrease second order harmonic
            v2 -=0.01;
            break; 
        case ('g'):// f key increase second order harmonic
            v1 +=0.01;
            break; 
        case ('b'):// g key decrease second order harmonic
            v1 -=0.01;
            break; 
            
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char **argv, const GLuint &vbo)
{
    if (!d_vbo_buffer)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

        // map buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        double *data = (double *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

        // check result
        if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
        {
            // write file for regression test
            sdkWriteFile<double>("./data/regression.dat",
                                data, Npar * 3, 0.0, false);
        }

        // unmap GL buffer object
        if (!glUnmapBuffer(GL_ARRAY_BUFFER))
        {
            fprintf(stderr, "Unmap buffer failed.\n");
            fflush(stderr);
        }

        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsWriteDiscard));

        SDK_CHECK_ERROR_GL();
    }
}
