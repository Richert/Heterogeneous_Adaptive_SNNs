!------------------------------------------------------------------------------
!------------------------------------------------------------------------------
!   qif :  QIF multi-population model with synaptic depression
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      INTEGER :: n, M
      DOUBLE PRECISION :: R((NDIM)/4),V((NDIM)/4),S((NDIM)/4),A((NDIM)/4)
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)
      DOUBLE PRECISION I,J,D,tau_s,tau_a,A0,PI,RM,DS

       I = PAR(1)
       J = PAR(2)
       D = PAR(3)
       tau_s = PAR(4)
       tau_a = PAR(5)
       A0 = PAR(6)
       PI = 4*ATAN(1.0D0)
       M = (NDIM)/4

       RM = 0
       do n=1,M
         R(n) = U(n)
         V(n) = U(n+M)
         S(n) = U(n+2*M)
         A(n) = U(n+3*M)
       end do

       RM = 0
       do n=1,M
         RM = RM + S(n)
       end do
       RM = RM * J/M
       DS = D/(PI*2.0*M)

       do n=1,M
         F(n) = DS + 2.0*R(n)*V(n)
         F(n+M) = V(n)*V(n) + RM + I + D*((n-1.0)/(M-1.0) - 0.5) - PI*PI*R(n)*R(n)
         F(n+2*M) = (A(n)*R(n) - S(n)) / tau_s
         F(n+3*M) = (A0-A(n))/tau_a + A0*(1-A(n))*R(n)
       end do

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      INTEGER :: n,M
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T
      DOUBLE PRECISION I,J,D,tau_s,tau_a,A0,TPI

       I = -10.0
       J = 0.0
       D = 2.0
       tau_s = 0.5
       tau_a = 20.0
       A0 = 1.0

       PAR(1)=I
       PAR(2)=J
       PAR(3)=D
       PAR(4)=tau_s
       PAR(5)=tau_a
       PAR(6)=A0
       TPI=8.0*ATAN(1.0D0)
       M =(NDIM)/4

       do n=1,M
         U(n) = 0.0
         U(n+M) = -SQRT(-(I-D+(n-0.5)*(2.0*D/M)))
         U(n+2*M) = 0.0
         U(n+3*M) = A0
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