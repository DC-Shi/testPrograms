! This code is to calculate C = 2*A + B
! The code compiles to CUDA and OpenACC
! It contains two types of computation: i,j,k and k,j,i
! additional two cases: 4 manually unroll and non-unroll
! We want to compare the difference in perf and SASS 

PROGRAM Loop_OpenACC_CUDA

#ifdef _OPENACC
    use openacc
#endif
    implicit none

    integer, parameter :: N = 512
    integer :: i, j, k, error
    double precision, dimension(N,N,N) :: A, B, C
    #ifdef _CUDA
      !@cuf attributes(managed) :: A,B,C
    #endif

    ! init A,B,C
#ifdef _CUDA
    !$cuf kernel do(3) <<< *, 128 >>>
#endif
#ifdef _OPENACC
    !$acc enter data copyin(a,b,c)
    !$acc parallel loop collapse(3) default(present)
#endif

    #ifdef ORDERKJI
    DO k = 1,N
      DO j = 1,N
        DO i = 1,N
    #else
    DO i = 1,N
      DO j = 1,N
        DO k = 1,N
    #endif
          A(i,j,k) = i+j+k
          B(i,j,k) = i*j/k
          C(i,j,k) = 0
        ENDDO
      ENDDO
    ENDDO
#ifdef _OPENACC
    !$acc end parallel
#endif


    ! compute C=2*A+B
#ifdef _CUDA
    !$cuf kernel do(3) <<< *, 128 >>>
#endif
#ifdef _OPENACC
    !$acc parallel loop collapse(3) default(present)
#endif

    #ifdef ORDERKJI
    DO k = 1,N
      DO j = 1,N
        DO i = 1,N
    #else
    DO i = 1,N
      DO j = 1,N
        DO k = 1,N
    #endif
          C(i,j,k) = A(i,j,k) + B(i,j,k)
        ENDDO
      ENDDO
    ENDDO
#ifdef _OPENACC
    !$acc end parallel
#endif

#ifdef _CUDA
    call cudaDeviceSynchronize()
#endif
#ifdef _OPENACC
    !$acc exit data copyout(C)
#endif

    print *, 'Finished calculating'
    
    error = 0
    DO i = 1,N
      DO j = 1,N
        DO k = 1,N
          if ( C(i,j,k) .ne. i+j+k+i*j/k ) then
            error = error + 1
          endif
        ENDDO
      ENDDO
    ENDDO
    print *, 'Finished checking'

    if ( error .eq. 0 ) then
      print *,'  Good'
    else
      print '(2x,"Bad, ",i6,"errors")', error
    endif

END

