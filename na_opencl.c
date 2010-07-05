/*
  na_opencl.c
  Numerical Array Extention on OpenCL for Ruby
    (C) Copyright 2010 by Kazuyuki TANIMURA

  This program is free software.
  You can distribute/modify this program
  under the same terms as Ruby itself.
  NO WARRANTY.
*/
#if (defined(HAVE_OPENCL_OPENCL_H) || defined(HAVE_CL_CL_H))
#include <stdio.h>
#include "ruby.h"
#include "narray.h"
#include "narray_local.h"

#ifdef HAVE_OPENCL_OPENCL_H
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "na_opencl.h"
#define MAX_SOURCE_SIZE (0x100000)

/* global variables */
cl_device_id device_id = NULL;
cl_context context = NULL;

void
 na_opencl_do_binary(cl_command_queue command_queue, cl_kernel kernel, size_t global_item_size, cl_mem O_buf, cl_mem A_buf, int i1, cl_mem B_buf, int i2, cl_mem C_buf, int i3)
{
  cl_int ret;

  /* set OpenCL kernel arguments */
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&O_buf);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A_buf);
  ret = clSetKernelArg(kernel, 2, sizeof(int),    (void *)&i1);
  ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&B_buf);
  ret = clSetKernelArg(kernel, 4, sizeof(int),    (void *)&i2);
  ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&C_buf);
  ret = clSetKernelArg(kernel, 6, sizeof(int),    (void *)&i3);

  /* execute OpenCL kernel */
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL); //let OpenCL decide the local item size by feeding NULL to 6th arg
  if (ret != CL_SUCCESS) {
    rb_raise(rb_eRuntimeError, "Failed executing kernel \n");
  }
}

void
 na_opencl_allocate( int nd, char *p1, char *p2, char *p3, struct slice *s1, struct slice *s2, struct slice *s3, void* kernel_func )
{
  int i;
  int ps1 = s1[0].pstep;
  int ps2 = s2[0].pstep;
  int ps3 = s3[0].pstep;
  int *si;

  si = ALLOCA_N(int,nd);
  i  = nd;
  s1[i].p = p1;
  s2[i].p = p2;
  s3[i].p = p3;
///////////////////////////////////////////////////
  cl_mem O_buffer = NULL;
  cl_mem A_buffer = NULL;
  cl_mem B_buffer = NULL;
  cl_mem C_buffer = NULL;
  cl_command_queue command_queue = NULL;
  cl_int ret;

  /* create OpenCL command queue */
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
///////////////////////////////////////////////////
  for(;;) {
    /* set pointers */
    while (i > 0) {
      --i;
      s3[i].p = s3[i].pbeg + s3[i+1].p;
      s2[i].p = s2[i].pbeg + s2[i+1].p;
      s1[i].p = s1[i].pbeg + s1[i+1].p;
      si[i] = s1[i].n;
    }
    /* rank 0 loop */
///////////////////////////////////////////////////
    /* create memory buffer */
    O_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, s2[0].n*ps1*sizeof(char), NULL, &ret);
    A_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, s2[0].n*ps1*sizeof(char), NULL, &ret);
    B_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, s2[0].n*ps2*sizeof(char), NULL, &ret);
    C_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, s2[0].n*ps3*sizeof(char), NULL, &ret);

    /* write memory buffer */
    ret = clEnqueueWriteBuffer(command_queue, A_buffer, CL_TRUE, 0, s2[0].n*ps1*sizeof(char),s1[0].p, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, B_buffer, CL_TRUE, 0, s2[0].n*ps2*sizeof(char),s2[0].p, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, C_buffer, CL_TRUE, 0, s2[0].n*ps3*sizeof(char),s3[0].p, 0, NULL, NULL);

    //na_opencl_do_binary(command_queue, kernel_func, O_buffer, A_buffer, B_buffer, C_buffer, mem_size);
    na_opencl_do_binary(command_queue, kernel_func, s2[0].n, O_buffer, A_buffer, ps1, B_buffer, ps2, C_buffer, ps3);

    /* read memory buffer */
    ret = clEnqueueReadBuffer(command_queue, O_buffer, CL_TRUE, 0, s2[0].n*ps1*sizeof(char),s1[0].p, 0, NULL, NULL);

    /* run commands in queue */
    ret = clFlush(command_queue);
    /* make sure all commands in queue is done */
    ret = clFinish(command_queue);
    /* releasing OpenCL objects */
    ret = clReleaseMemObject(O_buffer);
    ret = clReleaseMemObject(A_buffer);
    ret = clReleaseMemObject(B_buffer);
    ret = clReleaseMemObject(C_buffer);
///////////////////////////////////////////////////
    /* rank up */
    do {
      if ( ++i >= nd ) {
///////////////////////////////////////////////////
        ret = clReleaseCommandQueue(command_queue);
///////////////////////////////////////////////////
        return;
      }
    } while ( --si[i] == 0 );
    /* next point */
    s1[i].p += s1[i].pstep;
    s2[i].p += s2[i].pstep;
    s3[i].p += s3[i].pstep;
  }
} 

void
 Init_na_opencl()
{
  //cl_device_id device_id = NULL;
  //cl_context context = NULL;
  cl_program program = NULL;
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;

  char fileName[] = KERNEL_SRC_FILE;
  FILE *fp;
  char *kernel_src_code;
  size_t kernel_src_size;

  /* load kernel source code */
  fp = fopen(fileName, "r");
  if (!fp) {
    rb_raise(rb_eIOError, "Failed loading %s\n", fileName);
  }
  kernel_src_code = (char*)malloc(MAX_SOURCE_SIZE);
  kernel_src_size = fread( kernel_src_code, 1, MAX_SOURCE_SIZE, fp);
  fclose( fp );

  /* get platform device info */
  clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

  /* create OpenCL context */
  context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

  /* create kernel program from the kernel source code */
  program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src_code, (const size_t *)&kernel_src_size, &ret);
  free(kernel_src_code);

  /* build the kernel program */
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (ret != CL_SUCCESS) {
    size_t len;
    char log[2048];
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log), log, &len);
    rb_raise(rb_eRuntimeError, "Failed building %s\n", fileName);
  }

  /* create OpenCL kernels */
  CREATE_OPENCL_KERNELS(program, ret);

/////////////////////////////////////////////////////
//  /* releasing OpenCL objetcs */
//  ret = clReleaseKernel((cl_kernel)AddBFuncs[NA_LINT]); //for dev
//  ret = clReleaseKernel((cl_kernel)SbtBFuncs[NA_LINT]); //for dev
//  ret = clReleaseKernel((cl_kernel)MulBFuncs[NA_LINT]); //for dev
//  ret = clReleaseKernel((cl_kernel)DivBFuncs[NA_LINT]); //for dev
//  ret = clReleaseKernel((cl_kernel)ModBFuncs[NA_LINT]); //for dev
//  ret = clReleaseKernel((cl_kernel)MulAddFuncs[NA_LINT]); //for dev
//  ret = clReleaseKernel((cl_kernel)MulSbtFuncs[NA_LINT]); //for dev
//  ret = clReleaseProgram(program); //for dev
//  ret = clReleaseContext(context); //for dev
/////////////////////////////////////////////////////

}
#endif
