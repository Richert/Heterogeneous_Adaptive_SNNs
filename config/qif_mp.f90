!------------------------------------------------------------------------------
!------------------------------------------------------------------------------
!   qif :  QIF multi-population model
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      INTEGER :: n, M
      DOUBLE PRECISION :: R((NDIM)/3),V((NDIM)/3),S((NDIM)/3)
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)
      DOUBLE PRECISION I,J,D,tau_s,D2,PI,RM,I_m,I_step

       I = PAR(1)
       J = PAR(2)
       D = PAR(3)
       tau_s = PAR(4)
       M = (NDIM)/3
       D2 = D/(2*M)
       PI = 4*ATAN(1.0D0)
       I_m = -0.5
       I_step = 1.0/M

       do n=1,M
         R(n) = U(n)
         V(n) = U(n+M)
         S(n) = U(n+2*M)
       end do
       RM = 0
       do n=1,M
         RM = RM + S(n)
       end do

       do n=1,M
         F(n) = D2/PI + 2.0*R(n)*V(n)
         F(n+M) = V(n)*V(n) + J*RM/M - PI*PI*R(n)*R(n) + I + D*I_m
         F(n+2*M) = (R(n) - S(n)) / tau_s
         I_m = I_m + I_step
       end do

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      INTEGER :: n,M
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T
      DOUBLE PRECISION I,J,D,tau_s,TPI,I_m,I_step

       I = -3.0
       J = 0.0
       D = 1.0
       tau_s = 1.0

       PAR(1)=I
       PAR(2)=J
       PAR(3)=D
       PAR(4)=tau_s
       TPI=8.0*ATAN(1.0D0)
       M =(NDIM)/3
       I_m = -0.5
       I_step = 1.0/M

       do n=1,M
         U(n) = 0.0
         U(n+M) = -SQRT(-(I+D*I_m))
         U(n+2*M) = 0.0
         I_m = I_m + I_step
       end do

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS