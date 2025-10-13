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
      DOUBLE PRECISION I,J,D,tau_s,tau_a,A0,kappa,PI,RM

       I = PAR(1)
       J = PAR(2)
       D = PAR(3)
       tau_s = PAR(4)
       tau_a = PAR(5)
       A0 = PAR(6)
       kappa = PAR(7)
       PI = 4*ATAN(1.0D0)
       M = (NDIM)/4

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

       do n=1,M
         F(n) = D*(TAN(0.5*PI*(2*n-M-0.5)/(M+1))-TAN(0.5*PI*(2*n-M-1.5)/(M+1)))/PI + 2.0*R(n)*V(n)
         F(n+M) = V(n)*V(n) + J*RM/M - PI*PI*R(n)*R(n) + I + D*TAN(0.5*PI*(2*n-M-1)/(M+1))
         F(n+2*M) = (A(n)*R(n) - S(n)) / tau_s
         F(n+3*M) = (1-A(n))/tau_a + (A0-A(n))*kappa*R(n)
       end do

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      INTEGER :: n,M
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION I,J,D,tau_s,tau_a,A0,kappa,TPI

       I = -3.0
       J = 15.0
       D = 0.5
       tau_s = 1.0
       tau_a = 20.0
       A0 = 0.0
       kappa = 0.01

       PAR(1)=I
       PAR(2)=J
       PAR(3)=D
       PAR(4)=tau_s
       PAR(5)=tau_a
       PAR(6)=A0
       PAR(7)=kappa
       TPI=8.0*ATAN(1.0D0)
       M =(NDIM)/4

       do n=1,M
         U(n) = 0.0
         U(n+M) = -SQRT(-(I-D+(n-0.5)*(2.0*D/M)))
         U(n+2*M) = 0.0
         U(n+3*M) = 1.0
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