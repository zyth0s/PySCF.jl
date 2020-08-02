
#########################################################################
# Project #8: DIIS extrapolation for the SCF procedure.
# - P. Pulay, Chem. Phys. Lett. 73, 393-398 (1980).
# - P. Pulay, J. Comp. Chem. 3, 556-560 (1982).
# - T. P. Hamilton and P. Pulay, J. Chem. Phys. 84, 5728-5734 (1986).
# - C. David Sherrill. "Some comments on accellerating convergence of iterative
#      sequences using direct inversion of the iterative subspace (DIIS)".
#      Available at: vergil.chemistry.gatech.edu/notes/diis/diis.pdf. (1998)
#########################################################################

function diis(F, Fstack, DIISstack, s_minushalf, s, P, iteri)
   # orbital gradient in AO basis: F Di Si - S Di Fi
   # better choice:     (S^-0.5)' (F Di Si - S Di Fi) S^-0.5
   DIIS_residual = s * P * F
   DIIS_residual = DIIS_residual' - DIIS_residual
   DIIS_residual = s_minushalf * DIIS_residual * s_minushalf
   if iteri > 1
      push!(Fstack,F)
      push!(DIISstack, DIIS_residual)
   end
   ΔDIIS = norm(DIIS_residual)

   if iteri > 2
      if iteri > 15 # Limit the storage of DIIS arrays
         popfirst!(DIISstack)
         popfirst!(Fstack)
      end
      dim_B = length(Fstack) + 1 # 1 dim for the Lagrange multiplier
      B = zeros(dim_B,dim_B)
      B[end,:]   .= -1
      B[:,  end] .= -1
      B[end,end]  =  0
      for i in eachindex(Fstack), j in eachindex(Fstack)
         B[i,j] = dot(DIISstack[i], DIISstack[j]) # Gramian matrix
      end
      # Solve Lagrange equation of Pulay
      Pulay_rhs = zeros(dim_B)
      Pulay_rhs[end] = -1
      coef_Pulay = B \ Pulay_rhs
      F = zeros(size(F))
      for (Fi,c) in zip(Fstack,coef_Pulay[1:end-1]) # skip Lagrange mult.
         F += c * Fi
      end
   end
   F, Fstack, DIISstack
end
