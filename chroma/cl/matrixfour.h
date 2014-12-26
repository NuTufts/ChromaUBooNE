#ifndef __MATRIXFOUR_H__
#define __MATRIXFOUR_H__

struct MatrixFour
{
    float a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33 ;
};

__device__ MatrixFour
make_matrixfour(float a00, float a01, float a02, float a03,
	            float a10, float a11, float a12, float a13,
	            float a20, float a21, float a22, float a23,
	            float a30, float a31, float a32, float a33)
{
    MatrixFour m = {a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33};
    return m;
}

__device__ MatrixFour
make_matrixfour(const float4 &u1, const float4 &u2, const float4 &u3, const float4 &u4 )
{
    MatrixFour m = {u1.x, u2.x, u3.x, u4.x, u1.y, u2.y, u3.y, u4.y, u1.z, u2.z, u3.z, u4.z, u1.w, u2.w, u3.w, u4.w };
    return m;
}

__device__ MatrixFour
operator- (const MatrixFour &m)
{
    return make_matrixfour(-m.a00, -m.a01, -m.a02, -m.a03, 
		                   -m.a10, -m.a11, -m.a12, -m.a13, 
		                   -m.a20, -m.a21, -m.a22, -m.a23,
		                   -m.a30, -m.a31, -m.a32, -m.a33);
}

__device__ float4
operator* (const MatrixFour &m, const float4 &a)
{
    return make_float4(m.a00*a.x + m.a01*a.y + m.a02*a.z + m.a03*a.w,
		               m.a10*a.x + m.a11*a.y + m.a12*a.z + m.a13*a.w,
		               m.a20*a.x + m.a21*a.y + m.a22*a.z + m.a23*a.w,
		               m.a30*a.x + m.a31*a.y + m.a32*a.z + m.a33*a.w);
}

__device__ MatrixFour
operator+ (const MatrixFour &m, const MatrixFour &n)
{
    return make_matrixfour(m.a00+n.a00, m.a01+n.a01, m.a02+n.a02, m.a03+n.a03,
		                   m.a10+n.a10, m.a11+n.a11, m.a12+n.a12, m.a13+n.a13,
		                   m.a20+n.a20, m.a21+n.a21, m.a22+n.a22, m.a23+n.a23,
		                   m.a30+n.a30, m.a31+n.a31, m.a32+n.a32, m.a33+n.a33);
}

__device__ void
operator+= (MatrixFour &m, const MatrixFour &n)
{
    m.a00 += n.a00;
    m.a01 += n.a01;
    m.a02 += n.a02;
    m.a03 += n.a03;

    m.a10 += n.a10;
    m.a11 += n.a11;
    m.a12 += n.a12;
    m.a13 += n.a13;

    m.a20 += n.a20;
    m.a21 += n.a21;
    m.a22 += n.a22;
    m.a23 += n.a23;

    m.a30 += n.a30;
    m.a31 += n.a31;
    m.a32 += n.a32;
    m.a33 += n.a33;

}

__device__ MatrixFour
operator- (const MatrixFour &m, const MatrixFour &n)
{
    return make_matrixfour(m.a00-n.a00, m.a01-n.a01, m.a02-n.a02, m.a03-n.a03,
		                   m.a10-n.a10, m.a11-n.a11, m.a12-n.a12, m.a13-n.a13,
		                   m.a20-n.a20, m.a21-n.a21, m.a22-n.a22, m.a23-n.a23,
		                   m.a30-n.a30, m.a31-n.a31, m.a32-n.a32, m.a33-n.a33);
}

__device__ void
operator-= (MatrixFour &m, const MatrixFour& n)
{
    m.a00 -= n.a00;
    m.a01 -= n.a01;
    m.a02 -= n.a02;
    m.a03 -= n.a03;

    m.a10 -= n.a10;
    m.a11 -= n.a11;
    m.a12 -= n.a12;
    m.a13 -= n.a13;

    m.a20 -= n.a20;
    m.a21 -= n.a21;
    m.a22 -= n.a22;
    m.a23 -= n.a23;

    m.a30 -= n.a30;
    m.a31 -= n.a31;
    m.a32 -= n.a32;
    m.a33 -= n.a33;

}

__device__ MatrixFour
operator* (const MatrixFour &m, const MatrixFour &n)
{
    return make_matrixfour(
               m.a00*n.a00 + m.a01*n.a10 + m.a02*n.a20 + m.a03*n.a30,
		       m.a00*n.a01 + m.a01*n.a11 + m.a02*n.a21 + m.a03*n.a31,
		       m.a00*n.a02 + m.a01*n.a12 + m.a02*n.a22 + m.a03*n.a32,
		       m.a00*n.a03 + m.a01*n.a13 + m.a02*n.a23 + m.a03*n.a33,
		       m.a10*n.a00 + m.a11*n.a10 + m.a12*n.a20 + m.a13*n.a30,
		       m.a10*n.a01 + m.a11*n.a11 + m.a12*n.a21 + m.a13*n.a31,
		       m.a10*n.a02 + m.a11*n.a12 + m.a12*n.a22 + m.a13*n.a32,
		       m.a10*n.a03 + m.a11*n.a13 + m.a12*n.a23 + m.a13*n.a33,
		       m.a20*n.a00 + m.a21*n.a10 + m.a22*n.a20 + m.a23*n.a30,
		       m.a20*n.a01 + m.a21*n.a11 + m.a22*n.a21 + m.a23*n.a31,
		       m.a20*n.a02 + m.a21*n.a12 + m.a22*n.a22 + m.a23*n.a32,
		       m.a20*n.a03 + m.a21*n.a13 + m.a22*n.a23 + m.a23*n.a33,
		       m.a30*n.a00 + m.a31*n.a10 + m.a32*n.a20 + m.a33*n.a30,
		       m.a30*n.a01 + m.a31*n.a11 + m.a32*n.a21 + m.a33*n.a31,
		       m.a30*n.a02 + m.a31*n.a12 + m.a32*n.a22 + m.a33*n.a32,
		       m.a30*n.a03 + m.a31*n.a13 + m.a32*n.a23 + m.a33*n.a33);
}

__device__ MatrixFour
operator+ (const MatrixFour &m, const float &c)
{
    return make_matrixfour(m.a00+c, m.a01+c, m.a02+c, m.a03+c,
		                   m.a10+c, m.a11+c, m.a12+c, m.a13+c,
		                   m.a20+c, m.a21+c, m.a22+c, m.a23+c,
		                   m.a30+c, m.a31+c, m.a32+c, m.a33+c);
}

__device__ void
operator+= (MatrixFour &m, const float &c)
{
    m.a00 += c;
    m.a01 += c;
    m.a02 += c;
    m.a03 += c;

    m.a10 += c;
    m.a11 += c;
    m.a12 += c;
    m.a13 += c;

    m.a20 += c;
    m.a21 += c;
    m.a22 += c;
    m.a23 += c;

    m.a30 += c;
    m.a31 += c;
    m.a32 += c;
    m.a33 += c;

}

__device__ MatrixFour
operator+ (const float &c, const MatrixFour &m)
{
    return make_matrixfour(c+m.a00, c+m.a01, c+m.a02, c+m.a03,
		                   c+m.a10, c+m.a11, c+m.a12, c+m.a13,
		                   c+m.a20, c+m.a21, c+m.a22, c+m.a23,
		                   c+m.a30, c+m.a31, c+m.a32, c+m.a33);
}

__device__ MatrixFour
operator- (const MatrixFour &m, const float &c)
{
    return make_matrixfour(m.a00-c, m.a01-c, m.a02-c, m.a03-c,
		                   m.a10-c, m.a11-c, m.a12-c, m.a13-c,
		                   m.a20-c, m.a21-c, m.a22-c, m.a23-c,
		                   m.a30-c, m.a31-c, m.a32-c, m.a33-c);
}

__device__ void
operator-= (MatrixFour &m, const float &c)
{
    m.a00 -= c;
    m.a01 -= c;
    m.a02 -= c;
    m.a03 -= c;

    m.a10 -= c;
    m.a11 -= c;
    m.a12 -= c;
    m.a13 -= c;

    m.a20 -= c;
    m.a21 -= c;
    m.a22 -= c;
    m.a23 -= c;

    m.a30 -= c;
    m.a31 -= c;
    m.a32 -= c;
    m.a33 -= c;


}

__device__ MatrixFour
operator- (const float &c, const MatrixFour &m)
{
    return make_matrixfour(c-m.a00, c-m.a01, c-m.a02, c-m.a03,
		                   c-m.a10, c-m.a11, c-m.a12, c-m.a13,
		                   c-m.a20, c-m.a21, c-m.a22, c-m.a23,
		                   c-m.a30, c-m.a31, c-m.a32, c-m.a33);
}

__device__ MatrixFour
operator* (const MatrixFour &m, const float &c)
{
    return make_matrixfour(m.a00*c, m.a01*c, m.a02*c, m.a03*c,
		                   m.a10*c, m.a11*c, m.a12*c, m.a13*c, 
		                   m.a20*c, m.a21*c, m.a22*c, m.a23*c,
		                   m.a30*c, m.a31*c, m.a32*c, m.a33*c);
}

__device__ void
operator*= (MatrixFour &m, const float &c)
{
    m.a00 *= c;
    m.a01 *= c;
    m.a02 *= c;
    m.a03 *= c;

    m.a10 *= c;
    m.a11 *= c;
    m.a12 *= c;
    m.a13 *= c;

    m.a20 *= c;
    m.a21 *= c;
    m.a22 *= c;
    m.a23 *= c;

    m.a30 *= c;
    m.a31 *= c;
    m.a32 *= c;
    m.a33 *= c;

}

__device__ MatrixFour
operator* (const float &c, const MatrixFour &m)
{
    return make_matrixfour(c*m.a00, c*m.a01, c*m.a02, c*m.a03,
		                   c*m.a10, c*m.a11, c*m.a12, c*m.a13,
		                   c*m.a20, c*m.a21, c*m.a22, c*m.a23,
		                   c*m.a30, c*m.a31, c*m.a32, c*m.a33);
}

__device__ MatrixFour
operator/ (const MatrixFour &m, const float &c)
{
    return make_matrixfour(m.a00/c, m.a01/c, m.a02/c, m.a03/c,
		                   m.a10/c, m.a11/c, m.a12/c, m.a13/c,
		                   m.a20/c, m.a21/c, m.a22/c, m.a23/c,
		                   m.a30/c, m.a31/c, m.a32/c, m.a33/c);
}

__device__ void
operator/= (MatrixFour &m, const float &c)
{
    m.a00 /= c;
    m.a01 /= c;
    m.a02 /= c;
    m.a03 /= c;

    m.a10 /= c;
    m.a11 /= c;
    m.a12 /= c;
    m.a13 /= c;

    m.a20 /= c;
    m.a21 /= c;
    m.a22 /= c;
    m.a23 /= c;

    m.a30 /= c;
    m.a31 /= c;
    m.a32 /= c;
    m.a33 /= c;

}

__device__ MatrixFour
operator/ (const float &c, const MatrixFour &m)
{
    return make_matrixfour(c/m.a00, c/m.a01, c/m.a02, c/m.a03,
		                   c/m.a10, c/m.a11, c/m.a12, c/m.a13,
		                   c/m.a20, c/m.a21, c/m.a22, c/m.a23,
		                   c/m.a30, c/m.a31, c/m.a32, c/m.a33);
}


__device__ float
det(const MatrixFour &m)
{
  // http://www.euclideanspace.com/maths/algebra/matrix/functions/determinant/fourD/index.htm
  return 
      m.a03 * m.a12 * m.a21 * m.a30 - m.a02 * m.a13 * m.a21 * m.a30 - m.a03 * m.a11 * m.a22 * m.a30 + m.a01 * m.a13 * m.a22 * m.a30 +
      m.a02 * m.a11 * m.a23 * m.a30 - m.a01 * m.a12 * m.a23 * m.a30 - m.a03 * m.a12 * m.a20 * m.a31 + m.a02 * m.a13 * m.a20 * m.a31 +
      m.a03 * m.a10 * m.a22 * m.a31 - m.a00 * m.a13 * m.a22 * m.a31 - m.a02 * m.a10 * m.a23 * m.a31 + m.a00 * m.a12 * m.a23 * m.a31 +
      m.a03 * m.a11 * m.a20 * m.a32 - m.a01 * m.a13 * m.a20 * m.a32 - m.a03 * m.a10 * m.a21 * m.a32 + m.a00 * m.a13 * m.a21 * m.a32 +
      m.a01 * m.a10 * m.a23 * m.a32 - m.a00 * m.a11 * m.a23 * m.a32 - m.a02 * m.a11 * m.a20 * m.a33 + m.a01 * m.a12 * m.a20 * m.a33 +
      m.a02 * m.a10 * m.a21 * m.a33 - m.a00 * m.a12 * m.a21 * m.a33 - m.a01 * m.a10 * m.a22 * m.a33 + m.a00 * m.a11 * m.a22 * m.a33 ;
}



__device__ MatrixFour
inv(const MatrixFour &m)
{
   // http://www.euclideanspace.com/maths/algebra/matrix/functions/inverse/fourD/index.htm
   return make_matrixfour(
         m.a12*m.a23*m.a31 - m.a13*m.a22*m.a31 + m.a13*m.a21*m.a32 - m.a11*m.a23*m.a32 - m.a12*m.a21*m.a33 + m.a11*m.a22*m.a33,
         m.a03*m.a22*m.a31 - m.a02*m.a23*m.a31 - m.a03*m.a21*m.a32 + m.a01*m.a23*m.a32 + m.a02*m.a21*m.a33 - m.a01*m.a22*m.a33,
         m.a02*m.a13*m.a31 - m.a03*m.a12*m.a31 + m.a03*m.a11*m.a32 - m.a01*m.a13*m.a32 - m.a02*m.a11*m.a33 + m.a01*m.a12*m.a33,
         m.a03*m.a12*m.a21 - m.a02*m.a13*m.a21 - m.a03*m.a11*m.a22 + m.a01*m.a13*m.a22 + m.a02*m.a11*m.a23 - m.a01*m.a12*m.a23,
         m.a13*m.a22*m.a30 - m.a12*m.a23*m.a30 - m.a13*m.a20*m.a32 + m.a10*m.a23*m.a32 + m.a12*m.a20*m.a33 - m.a10*m.a22*m.a33,
         m.a02*m.a23*m.a30 - m.a03*m.a22*m.a30 + m.a03*m.a20*m.a32 - m.a00*m.a23*m.a32 - m.a02*m.a20*m.a33 + m.a00*m.a22*m.a33,
         m.a03*m.a12*m.a30 - m.a02*m.a13*m.a30 - m.a03*m.a10*m.a32 + m.a00*m.a13*m.a32 + m.a02*m.a10*m.a33 - m.a00*m.a12*m.a33,
         m.a02*m.a13*m.a20 - m.a03*m.a12*m.a20 + m.a03*m.a10*m.a22 - m.a00*m.a13*m.a22 - m.a02*m.a10*m.a23 + m.a00*m.a12*m.a23,
         m.a11*m.a23*m.a30 - m.a13*m.a21*m.a30 + m.a13*m.a20*m.a31 - m.a10*m.a23*m.a31 - m.a11*m.a20*m.a33 + m.a10*m.a21*m.a33,
         m.a03*m.a21*m.a30 - m.a01*m.a23*m.a30 - m.a03*m.a20*m.a31 + m.a00*m.a23*m.a31 + m.a01*m.a20*m.a33 - m.a00*m.a21*m.a33,
         m.a01*m.a13*m.a30 - m.a03*m.a11*m.a30 + m.a03*m.a10*m.a31 - m.a00*m.a13*m.a31 - m.a01*m.a10*m.a33 + m.a00*m.a11*m.a33,
         m.a03*m.a11*m.a20 - m.a01*m.a13*m.a20 - m.a03*m.a10*m.a21 + m.a00*m.a13*m.a21 + m.a01*m.a10*m.a23 - m.a00*m.a11*m.a23,
         m.a12*m.a21*m.a30 - m.a11*m.a22*m.a30 - m.a12*m.a20*m.a31 + m.a10*m.a22*m.a31 + m.a11*m.a20*m.a32 - m.a10*m.a21*m.a32,
         m.a01*m.a22*m.a30 - m.a02*m.a21*m.a30 + m.a02*m.a20*m.a31 - m.a00*m.a22*m.a31 - m.a01*m.a20*m.a32 + m.a00*m.a21*m.a32,
         m.a02*m.a11*m.a30 - m.a01*m.a12*m.a30 - m.a02*m.a10*m.a31 + m.a00*m.a12*m.a31 + m.a01*m.a10*m.a32 - m.a00*m.a11*m.a32,
         m.a01*m.a12*m.a20 - m.a02*m.a11*m.a20 + m.a02*m.a10*m.a21 - m.a00*m.a12*m.a21 - m.a01*m.a10*m.a22 + m.a00*m.a11*m.a22)/det(m);
}


__device__ MatrixFour
outer(const float4 &a, const float4 &b)
{
    return make_matrixfour( a.x*b.x, a.x*b.y, a.x*b.z, a.x*b.w, 
		                    a.y*b.x, a.y*b.y, a.y*b.z, a.y*b.w,
		                    a.z*b.x, a.z*b.y, a.z*b.z, a.z*b.w,
		                    a.w*b.x, a.w*b.y, a.w*b.z, a.w*b.w);
}

#endif
