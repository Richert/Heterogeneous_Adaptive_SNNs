!------------------------------------------------------------------------------
!------------------------------------------------------------------------------
!   qif :  QIF multi-population model with synaptic depression
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      INTEGER :: n, n2, M
      DOUBLE PRECISION :: R((NDIM)/16),V((NDIM)/16),S((NDIM)/16),A((NDIM)/16)
      DOUBLE PRECISION :: W((NDIM*NDIM)/(16*16)), RM((NDIM)/16)
      DOUBLE PRECISION :: UP((NDIM)/16), UD((NDIM)/16)
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)
      DOUBLE PRECISION I,J,D,tau_s,tau_a,kappa,PI,DS,a_p,a_d,tau_p,tau_d,b
      DOUBLE PRECISION XP,XD,W2,tau,tau_r,a0,a_r

       I = PAR(1)
       J = PAR(2)
       D = PAR(3)
       tau_s = PAR(4)
       tau_a = PAR(5)
       kappa = PAR(6)
       tau = PAR(7)
       tau_r = PAR(8)
       a0 = PAR(9)
       a_r = PAR(15)
       b = PAR(16)
       PI = 4*ATAN(1.0D0)
       M = (NDIM)/16
       DS = D/(PI*2.0*M)
       a_p = a0*a_r
       a_d = a0/a_r
       tau_p = tau
       tau_d = tau*tau_r

       do n=1,M
         R(n) = U(n)
         V(n) = U(n+M)
         S(n) = U(n+2*M)
         A(n) = U(n+3*M)
         UP(n) = U(n+4*M)
         UD(n) = U(n+5*M)
         do n2=1,M
           W((n-1)*M+n2) = U(6*M+(n-1)*M+n2)
         end do
       end do

       do n=1,M
         RM(n) = 0.0
       end do
       do n=1,M
         do n2=1,M
           RM(n) = RM(n) + W((n-1)*M+n2)*S(n2)
         end do
       end do

       do n=1,M
         F(n) = DS + 2.0*R(n)*V(n)
         F(n+M) = V(n)*V(n) + RM(n)*J/(0.5*M) + I + D*((n-1.0)/(M-1.0) - 0.5) - PI*PI*R(n)*R(n)
         F(n+2*M) = A(n)*R(n) - S(n)/tau_s
         F(n+3*M) = (1.0-A(n))/tau_a - kappa*A(n)*R(n)
         F(n+4*M) = R(n) - UP(n)/tau_p
         F(n+5*M) = R(n) - UD(n)/tau_d
         do n2=1,M
           XP = a_p*S(n)*UP(n2)
           XD = a_d*S(n2)*UD(n)
           W2 = W((n-1)*M+n2)
           F(6*M+(n-1)*M+n2) = b*((1.0-W2)*XP-W2*XD) + (1.0-b)*W2*(1.0-W2)*(XP-XD)
         end do
       end do

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      INTEGER :: n,n2,M
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T
      DOUBLE PRECISION I,J,D,tau_s,tau_a,kappa,TPI
      DOUBLE PRECISION tau,tau_r,a0,a_r,b

       I = -10.0
       J = 0.0
       D = 2.0
       tau_s = 0.5
       tau_a = 20.0
       kappa = 0.0
       b = 0.5
       tau = 10.0
       tau_r = 1.0
       a0 = 0.001
       a_r = 1.0

       PAR(1)=I
       PAR(2)=J
       PAR(3)=D
       PAR(4)=tau_s
       PAR(5)=tau_a
       PAR(6)=kappa
       PAR(7)=tau
       PAR(8)=tau_r
       PAR(9)=a0
       PAR(15)=a_r
       PAR(16)=b
       TPI=8.0*ATAN(1.0D0)
       M =(NDIM)/16

       do n=1,M
         U(n) = 0.0
         U(n+M) = -SQRT(-(I-D+(n-0.5)*(2.0*D/M)))
         U(n+2*M) = 0.0
         U(n+3*M) = 1.0
         U(n+4*M) = 0.0
         U(n+5*M) = 0.0
         do n2=1,M
           U(6*M+(n-1)*M+n2) = 0.5
         end do
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