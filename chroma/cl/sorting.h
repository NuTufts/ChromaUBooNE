// No templates for OpenCL which uses C99.
// This will become a hot mess.
// Though, if this is pointer manipulation we can go with char* ...


void swap( void* a, void* b, size_t size );
void reverse(int n, void* array, size_t elemsize );
void piksrt( int n, __local float* arrf );
void piksrt_device( int n, float* arrf );
void piksrt2( int n, float* arrf, uint4* arrvec );
unsigned long searchsorted(unsigned long n, float *arr, const float*x);
void insert(unsigned long n, void *arr, unsigned long i, const void *x, size_t elemsize);
void add_sorted(unsigned long n, float *arr, const float *x);

/* template <class T> */
/* __device__ void */
/* swap(T &a, T &b) */
/* { */
/*     T tmp = a; */
/*     a = b; */
/*     b = tmp; */
/* } */
void swap( void* a, void* b, size_t size ) {
  char temp[size]; // allocates temp memory for element value
  memcpy(temp,b,size); // below copying referenced value, not the pointer!
  memcpy(b,a,size);   
  memcpy(a,temp,size);
}


/* template <class T> */
/* __device__ void */
/* reverse(int n, T *a) */
/* { */
/*     for (int i=0; i < n/2; i++) */
/* 	swap(a[i],a[n-1-i]); */
/* } */
void reverse(int n, void* array, size_t elemsize ) {
  for (int i=0; i<n/2; i++) {
    char* pelem1 = array + i;         // pointer to element 1 of array
    char* pelem2 = array + n - 1 -i;  // pointer to element 2 of array
    swap( pelem1, pelem2, elemsize ); // pass pointers to elements and size of object
  }
}

/* template <class T> */
/* __device__ void */
/* piksrt(int n, T *arr) */
/* { */
/*     int i,j; */
/*     T a; */

/*     for (j=1; j < n; j++) { */
/* 	a = arr[j]; */
/* 	i = j-1; */
/* 	while (i >= 0 && arr[i] > a) { */
/* 	    arr[i+1] = arr[i]; */
/* 	    i--; */
/* 	} */
/* 	arr[i+1] = a; */
/*     } */
/* } */
// no way to compare values. when don't know type. we have to explicitly create function!
void piksrt( int n, __local float* arrf ) {
  // example inputs
  //int distance_table_len = 0;
  //float distance_table[1000];

  int i, j;
  float atemp;

  for (j=1; j<n; j++) {
    atemp = *(arrf+j);
    i = j-1;
    while (i>=0 && *(arrf+i)>atemp) {
      *(arrf+i+1) = atemp;
      i--;
    }
    *(arrf+i+1) = atemp;
  }
}

void piksrt_device( int n, float* arrf ) {
  // example inputs
  //int distance_table_len = 0;
  //float distance_table[1000];

  int i, j;
  float atemp;

  for (j=1; j<n; j++) {
    atemp = *(arrf+j);
    i = j-1;
    while (i>=0 && *(arrf+i)>atemp) {
      *(arrf+i+1) = atemp;
      i--;
    }
    *(arrf+i+1) = atemp;
  }
}

/* template <class T, class U> */
/* __device__ void */
/* piksrt2(int n, T *arr, U *brr) */
/* { */
/*     int i,j; */
/*     T a; */
/*     U b; */

/*     for (j=1; j < n; j++) { */
/* 	a = arr[j]; */
/* 	b = brr[j]; */
/* 	i = j-1; */
/* 	while (i >= 0 && arr[i] > a) { */
/* 	    arr[i+1] = arr[i]; */
/* 	    brr[i+1] = brr[i]; */
/* 	    i--; */
/* 	} */
/* 	arr[i+1] = a; */
/* 	brr[i+1] = b; */
/*     } */
/* } */
void piksrt2( int n, float* arrf, uint4* arrvec ) {
  // example arrays
  //float distance[MAX_CHILD]; 
  //uint4 children[MAX_CHILD]; 
  int i, j;
  float atemp;
  uint4 btemp;

  for (j=1; j<n; j++) {
    atemp = *(arrf+j);
    btemp = *(arrvec+j);
    i = j-1;
    while (i>=0 && *(arrf+i)>atemp) {
      *(arrf+i+1) = atemp;
      *(arrvec+i+1) = btemp;
      i--;
    }
    *(arrf+i+1) = atemp;
    *(arrvec+i+1) = btemp;
  }
  
}

/* Returns the index in `arr` where `x` should be inserted in order to
   maintain order. If `n` equals one, return the index such that, when
   `x` is inserted, `arr` will be in ascending order.
*/
/* template <class T> */
/* __device__ unsigned long */
/* searchsorted(unsigned long n, T *arr, const T &x) */
/* { */
/*     unsigned long ju,jm,jl; */
/*     int ascnd; */

/*     jl = 0; */
/*     ju = n; */

/*     ascnd = (arr[n-1] >= arr[0]); */

/*     while (ju-jl > 1) { */
/* 	jm = (ju+jl) >> 1; */

/* 	if ((x > arr[jm]) == ascnd) */
/* 	    jl = jm; */
/* 	else */
/* 	    ju = jm; */
/*     } */

/*     if ((x <= arr[0]) == ascnd) */
/* 	return 0; */
/*     else */
/* 	return ju; */
/* } */
unsigned long searchsorted(unsigned long n, float *arr, const float *x) {
  // match by value of by pointer?  might not have a choice if untyped
  unsigned long ju,jm,jl;
  int ascnd; 
  jl = 0;
  ju = n;
  ascnd = ( arr[n-1] >= arr[0] );
  while (ju-jl > 1) {
    jm = (ju+jl) >> 1;
    if ((*x > arr[jm]) == ascnd)
      jl = jm;
    else
      ju = jm;
  }

  if ((*x <= arr[0]) == ascnd)
    return 0;
  else
    return ju;
}

/* template <class T> */
/* __device__ void */
/* insert(unsigned long n, T *arr, unsigned long i, const T &x) */
/* { */
/*     unsigned long j; */
/*     for (j=n-1; j > i; j--) */
/* 	arr[j] = arr[j-1]; */
/*     arr[i] = x; */
/* } */
void insert(unsigned long n, void *arr, unsigned long i, const void *x, size_t elemsize)
{
  unsigned long j;
  for (j=n-1; j > i; j--) {
    memcpy( arr+j, arr+j-1, elemsize );
  }
  memcpy( arr+i, x, elemsize );
}

/* template <class T> */
/* __device__ void */
/* add_sorted(unsigned long n, T *arr, const T &x) */
/* { */
/*     unsigned long i = searchsorted(n, arr, x); */

/*     if (i < n) */
/* 	insert(n, arr, i, x); */
/* } */
void add_sorted(unsigned long n, float *arr, const float *x)
{
  unsigned long i = searchsorted(n, arr, x);

  if (i < n)
    insert(n, arr, i, x, sizeof(float));
}
