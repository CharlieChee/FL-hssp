#!/usr/bin/python


load("hssp.sage")

from sage.modules.free_module_integer import IntegerLattice


#atm works only for k=-1
class multi_hssp:
  def __init__(self,n,l,kappa=-1,B_X=1,nx0_bits=None,B_A=None,recover_box_max=None,A_int_scale=None):
    self.n=n
    self.l=l
    self.kappa=kappa
    # Upper bound on absolute value of entries of X (entries sampled in [0, B_X])
    self.B_X=B_X
    # If set, sample A with entries in [0, B_A]; otherwise uniform mod x0 (legacy)
    self.B_A=B_A
    # If set (e.g. 1000): in gen_instance_fixed_X, A_ij = floor(U * A_int_scale) with U ~ Uniform[0, B_A+1)
    # (same spirit as X = floor(raw * 1000)); None keeps discrete uniform {0..B_A}
    self.A_int_scale=A_int_scale
    # Cap for recoverBox BFS rows; None -> max(4000, 800*n)
    self.recover_box_max=recover_box_max
    # If set, use this bit-length for x0 instead of the default iota formula (too small for large B_X)
    self.nx0_bits_override=nx0_bits
    
    
  def gen_instance(self,m=0): # m use to specify dimension of the sample 
    if m==0 and self.n % 2==1:
      m=self.n*(self.n+3)/2 # n is odd
    elif m==0 and self.n % 2 ==0:
      m=self.n*(self.n+4)/2 # n is even
    self.m=int(m)
    
    print ("n=",self.n,"l=",self.l,"m=",m),
    if self.kappa>-1: print ("kappa=",self.kappa),
    iota=0.035
    #self.nx0=int(2*iota*self.n^2+self.n*log(self.n,2))# this is lower bound for log(Q)    
    self.nx0=int(0.1*iota*self.n^2+self.n*log(self.n,2)) 
    
    if self.nx0_bits_override is not None:
      self.nx0=int(self.nx0_bits_override)
    print ("nx0=",self.nx0)
    # genParams returns the modulus x0, the weights vector a, the matrix x, the  sample vector b  
    self.x0,self.A,self.X,self.B=genParams_mat(self.n, self.m,self.nx0,self.l,self.B_X,self.B_A)
    
  def gen_instance_fixed_X(self,X,A=None):
    r"""
    Use a fixed integer matrix ``X`` (``m``×``n``); sample ``A`` unless ``A`` is given.
    If ``self.A_int_scale`` is set: ``A_ij = floor(U * A_int_scale)`` with ``U ~ Uniform[0, B_A+1)``
    (same quantization style as ``floor(pre_relu * 1000)`` for ``X``).
    Otherwise: uniform integers in ``[0,B_A]``.
    ``B = X A`` reduced mod ``x0``.
    """
    X=matrix(ZZ,X)
    self.m=int(X.nrows())
    if int(X.ncols())!=int(self.n):
      raise ValueError("gen_instance_fixed_X: X.ncols()=%s != n=%s" % (X.ncols(), self.n))
    print ("n=",self.n,"l=",self.l,"m=",self.m,"(fixed X)"),
    if self.kappa>-1: print ("kappa=",self.kappa),
    iota=0.035
    self.nx0=int(0.1*iota*self.n^2+self.n*log(self.n,2))
    if self.nx0_bits_override is not None:
      self.nx0=int(self.nx0_bits_override)
    print ("nx0=",self.nx0)
    x0=genpseudoprime(self.nx0)
    if A is None:
      if self.B_A is None:
        raise ValueError("gen_instance_fixed_X: set B_A to sample A, or pass A=")
      A=matrix(ZZ,self.n,self.l)
      if self.A_int_scale is not None:
        import numpy as _np
        sc=float(self.A_int_scale)
        for i in range(self.n):
          for j in range(self.l):
            u=_np.random.uniform(0, float(self.B_A) + 1.0)
            A[i,j]=int(_np.floor(u * sc))
        print ("A: floor(Uniform[0, B_A+1) * %s), B_A=%s" % (sc,self.B_A))
      else:
        for i in range(self.n):
          for j in range(self.l):
            A[i,j]=ZZ.random_element(self.B_A+1)
    else:
      A=matrix(ZZ,A)
      if A.nrows()!=self.n or A.ncols()!=self.l:
        raise ValueError("gen_instance_fixed_X: A must be %s×%s" % (self.n,self.l))
    B=X*A%x0
    self.x0,self.A,self.X,self.B=x0,A,X,B
    

# This is the the function to perform the attacks.    
#H is the instance to be attacked and alg is the algorithm to use :
#       **if alg='default' or alg='multi' runs the multivariate attack
#       **if alg='ns_original' runs the original Nguyen-Stern attack 
#       **if alg='ns' runs the Nguyen-Stern attack with the improved orthogonal lattice attack 
#       **if alg='statistical' runs the heuristic statistical attack (FastICA on Step1 MO;
#         matrix HLCP B=X*A is split per column: b_j = X*a_j mod x0, same X, l separate Step1+ICA runs)
    
def hssp_attack(H,alg='ns'): 
  n=H.n
  kappa=H.kappa
  l=H.l
  
  if rank(Matrix(Integers(H.x0),H.B[:l,:l]))<l:
     print('Non-invertible')
     MB=Matrix(Integers(0),H.X) 
     return MB
  if alg=='statistical':
     if kappa!=-1:
       print('statistical: only kappa=-1 is supported')
       return {'method':'statistical','error':'kappa'}
     import os
     os.makedirs("ICA", exist_ok=True)
     load("statistical.sage")
     try:
       from sklearn.decomposition import FastICA
     except ImportError as e:
       print('statistical requires scikit-learn: pip install scikit-learn')
       print(e)
       return {'method':'statistical','error':'sklearn'}
     import math as _math
     from time import time as _wall_time
     g=globals()
     g['FastICA']=FastICA
     g['math']=_math
     g['time']=_wall_time
     B_ica=H.B_X
     print('\nStatistical attack (FastICA): %s columns, B_ica=%s' % (l,B_ica))
     nfound_cols=[]
     nrafound_cols=[]
     for j in range(l):
       Aj=matrix(ZZ,n,1,[H.A[i,j] for i in range(n)])
       Bj=H.X*Aj%H.x0
       Bj=matrix(ZZ,Bj.nrows(),1,Bj.list())
       print('--- statistical column j=%s ---' % j)
       try:
         MOj,tt1,tt10,tt1O=Step1_Mat(H.n,H.kappa,H.x0,Aj,H.X,Bj,H.m)
       except Exception as e:
         print('  Step1_Mat exception:',type(e).__name__,e)
         nfound_cols.append(None)
         nrafound_cols.append(None)
         continue
       aj=vector(ZZ,[Aj[i,0] for i in range(n)])
       bj=vector(ZZ,[Bj[i,0] for i in range(H.m)])
       try:
         MOn,MO_lll=statistical_1(MOj,H.n,H.m,H.x0,H.X,aj,bj,H.kappa,B_ica)
         tica,tS2,nrafound,nfound=statistical_2(MOn,MO_lll,H.n,H.m,H.x0,H.X,aj,bj,H.kappa,B_ica)
       except Exception as e:
         print('  statistical_1/2 exception:',type(e).__name__,e)
         nfound_cols.append(None)
         nrafound_cols.append(None)
         continue
       nfound_cols.append(int(nfound))
       nrafound_cols.append(int(nrafound))
       print('  column %s: NFound(rows vs X.T)=%s  nrafound(coeffs in a)=%s / %s' % (j,nfound,nrafound,n))
     ok_nf=len(nfound_cols)==l and all((v is not None and v==n) for v in nfound_cols)
     ok_ra=len(nrafound_cols)==l and all((v is not None and v==n) for v in nrafound_cols)
     return {'method':'statistical','nfound_per_col':nfound_cols,'nrafound_per_col':nrafound_cols,
             'all_nfound_ge_n':ok_nf,'all_nrafound_ge_n':ok_ra}
  if alg=='ns_original':
     MO,tt1,tt10,tt1O= Step1_ori_Mat(H.n,H.kappa,H.x0,H.A,H.X,H.B,H.m) 
     MB,beta=Step2_BK_mat(MO,H.n,H.m,H.X,kappa=-1,B=H.B_X,recover_box_max=H.recover_box_max)
     return MB
  if alg=='ns':
     MO,tt1,tt10,tt1O= Step1_Mat(H.n,H.kappa,H.x0,H.A,H.X,H.B,H.m)  
     #MO,tt1,tt10,tt1O= Step1_Mat_extend(H.n,H.x0,H.A,H.X,H.B,H.m) 
     try:
        if IntegerLattice(MO) == IntegerLattice(H.X.T):
           print('IntegerLattice(MO)==IntegerLattice(X.T): True')
        else:
           print('IntegerLattice check: MO', MO.dimensions(), 'X.T', H.X.T.dimensions())
     except Exception as ex:
        print('IntegerLattice check skipped:', type(ex).__name__, ex)
     MB,beta=Step2_BK_mat(MO,H.n,H.m,H.X,kappa=-1,B=H.B_X,recover_box_max=H.recover_box_max)
     return MB
     
     
  if alg=='ns_ssp':       
      print ("Nguyen-Stern (Original) Attack")  
      bb=1/2

      unbalanced=(abs(n*bb-kappa)/n > 0.2)
      if kappa>0 and not unbalanced:
        print ('ns not applicable') 
        return
      t=cputime()
      
      MO,tt1,tt10,tt1O= Step1_original(H.n,H.kappa,H.x0,H.a,H.X,H.b,H.m)
      Y=ns_ssp(H,MO)
      tttot=cputime(t)
      return Y, H         
  if alg=='multi': 
     assert H.m>(n^2+n)/2, 'm too small'
     MO,tt1,tt10,tt1O= Step1_ori_Mat(H.n,H.kappa,H.x0,H.A,H.X,H.B,H.m) 
     if kappa>0:
        tei, tef, tsf,tt2, nrafound=bit_guessing(H.n,H.kappa,MO,H.x0,H.a,H.X,H.b,H.m) 
     else:
        tei, tef, tt2,nrafound,MB=eigen_mat(H.n,H.kappa,MO,H.x0,H.a,H.X,H.b,H.m) 
        return MB
     
  #print('MB:',MB)
  #print('dimsion',MB.ncols())
  
  
def genParams_mat(n=10,m=20,nx0=100 ,l=10,B_X=1,B_A=None):
  x0=genpseudoprime(nx0)


  # We generate the alpha_i's
  A=matrix(ZZ,n,l)
  if B_A is not None:
    for i in range(n):
      for j in range(l):
        A[i,j]=ZZ.random_element(B_A+1)
  else:
    for i in range(n):
      for j in range(l):
        A[i,j]=mod(ZZ.random_element(x0),x0)

  # The matrix X has m rows and must be of rank n; entries in [0, B_X] (B_X==1 -> {0,1})
  while True:
      X=Matrix(ZZ,m,n)
      for i in range(m):
        for j in range(n):
          X[i,j]=ZZ.random_element(B_X+1)
      print (X.rank())
      if X.rank()==n: break
  print (X.density().n())

  # We generate an instance of the HSSP: b=X*A
  B=X*A%x0
  return x0,A,X,B


def Step1_Mat_extend(n,x0,A,X,B,m): 
  
  M=orthoLattice_mat(B,x0)

  print ("Step 1"),
  t=cputime()
  M2=M.LLL()
  tt10=cputime(t)
  print ("LLL step1: %.1f" % cputime(t)),


  #commented by Jane
  #print('assert sum', sum([vi==0 and 1 or 0 for vi in M2*X]))
  #assert sum([vi==0 and 1 or 0 for vi in M2*X])==m-n
  
  MOrtho=M2[:m-n]

  #print
  #for i in range(m-n+1):
  #  print i,N(log(M2[i:i+1].norm(),2)),N(log(m^(n/(2*(m-n)))*sqrt((m-n)/17),2)+iota*m+nx0/(m-n)) #N(log(sqrt((m-n)*n)*(m/2)^(m/(2*(m-n))),2)+iota*m)
  
  print ("  log(Height,2)=",int(log(MOrtho.height(),2))),

  t2=cputime()
  ke=kernelLLL(MOrtho)
  tt1O=cputime(t2)
  print ("  Kernel: %.1f" % cputime(t2)),
  tt1=cputime(t)
  print ("  Total step1: %.1f" % tt1)

  return ke,tt1,tt10,tt1O

def Step1_Mat(n,v,x0,A,X,B,m): 
  
  M=orthoLattice_mat(B,x0)

  print ("Step 1"),
  t=cputime()
  M2=M.LLL()
  tt10=cputime(t)
  print ("LLL step1: %.1f" % cputime(t)),


  #commented by Jane
  #print('assert sum', sum([vi==0 and 1 or 0 for vi in M2*X]))
  #assert sum([vi==0 and 1 or 0 for vi in M2*X])==m-n
  
  MOrtho=M2[:m-n]

  #print
  #for i in range(m-n+1):
  #  print i,N(log(M2[i:i+1].norm(),2)),N(log(m^(n/(2*(m-n)))*sqrt((m-n)/17),2)+iota*m+nx0/(m-n)) #N(log(sqrt((m-n)*n)*(m/2)^(m/(2*(m-n))),2)+iota*m)
  
  print ("  log(Height,2)=",int(log(MOrtho.height(),2))),

  t2=cputime()
  ke=kernelLLL(MOrtho)
  tt1O=cputime(t2)
  print ("  Kernel: %.1f" % cputime(t2)),
  tt1=cputime(t)
  print ("  Total step1: %.1f" % tt1)

  return ke,tt1,tt10,tt1O
  
def Step1_ori_Mat(n,v,x0,A,X,B,m): 
  
  M=orthoLattice_mat(B,x0)

  print ("Step 1"),
  t=cputime()
  M2=M.LLL()
  tt10=cputime(t)
  print ("LLL step1: %.1f" % cputime(t)),


  #commented by Jane
  #print('assert sum', sum([vi==0 and 1 or 0 for vi in M2*X]))
  #assert sum([vi==0 and 1 or 0 for vi in M2*X])==m-n
  
  MOrtho=M2[:m-n]

  #print
  #for i in range(m-n+1):
  #  print i,N(log(M2[i:i+1].norm(),2)),N(log(m^(n/(2*(m-n)))*sqrt((m-n)/17),2)+iota*m+nx0/(m-n)) #N(log(sqrt((m-n)*n)*(m/2)^(m/(2*(m-n))),2)+iota*m)
  
  print ("  log(Height,2)=",int(log(MOrtho.height(),2))),

  t2=cputime()
  ke=kernelLLL(MOrtho)
  tt1O=cputime(t2)
  print ("  Kernel: %.1f" % cputime(t2)),
  tt1=cputime(t)
  print ("  Total step1: %.1f" % tt1)

  return ke,tt1,tt10,tt1O
  
  
  

from fpylll import BKZ

def recoverBox(M5, B, max_size=4000):
  r"""
  Like ``recoverBinary`` but for coefficients in ``[-B, B]``: BFS closure under ``row ± v``
  while staying in the box. Capped at ``max_size`` rows so large ``B`` cannot blow up memory.
  """
  def in_box(v):
    return all(-B <= vj <= B for vj in v)
  lv = [vector(v) for v in M5 if in_box(v)]
  seen = {tuple(v) for v in lv}
  n = M5.nrows()
  idx = 0
  while idx < len(lv) and len(lv) < max_size:
    v = lv[idx]
    for i in range(n):
      if len(lv) >= max_size:
        break
      for nv in (vector(M5[i]) - v, vector(M5[i]) + v):
        if len(lv) >= max_size:
          break
        if in_box(nv):
          t = tuple(nv)
          if t not in seen:
            seen.add(t)
            lv.append(vector(nv))
    idx += 1
  if len(lv) == 0:
    return matrix(ZZ, 0, M5.ncols())
  return matrix(lv)

def recoverBox_v2(M5, B, max_size=4000):
  r"""
  Improved recoverBox:
  1. basis vector +/- known vector (same as original)
  2. known vector +/- known vector (new: pairwise combinations)
  3. non-negative priority: after exhaustive search, prefer vectors in [0,B] (true X columns are non-negative)
  """
  def in_box(v):
    return all(-B <= vj <= B for vj in v)
  def is_nonneg(v):
    return all(0 <= vj <= B for vj in v)

  lv = [vector(v) for v in M5 if in_box(v)]
  seen = {tuple(v) for v in lv}
  n_basis = M5.nrows()

  # Phase 1: basis vector +/- known vector (same as original)
  idx = 0
  while idx < len(lv) and len(lv) < max_size:
    v = lv[idx]
    for i in range(n_basis):
      if len(lv) >= max_size:
        break
      for nv in (vector(M5[i]) - v, vector(M5[i]) + v):
        if len(lv) >= max_size:
          break
        if in_box(nv):
          t = tuple(nv)
          if t not in seen:
            seen.add(t)
            lv.append(vector(nv))
    idx += 1

  # Phase 2: pairwise combinations of known vectors (new)
  # Only combine non-negative vectors to limit search size
  nonneg_vecs = [v for v in lv if is_nonneg(v)]
  phase2_budget = max_size - len(lv)
  added_p2 = 0
  for i in range(len(nonneg_vecs)):
    if added_p2 >= phase2_budget:
      break
    for j in range(i+1, len(nonneg_vecs)):
      if added_p2 >= phase2_budget:
        break
      for nv in (nonneg_vecs[i] - nonneg_vecs[j], nonneg_vecs[j] - nonneg_vecs[i]):
        if added_p2 >= phase2_budget:
          break
        if in_box(nv):
          t = tuple(nv)
          if t not in seen:
            seen.add(t)
            lv.append(vector(nv))
            added_p2 += 1

  if len(lv) == 0:
    return matrix(ZZ, 0, M5.ncols())
  return matrix(lv)


def Step2_BK_mat(ke,n,m,X,kappa=-1,B=1,recover_box_max=None):
  #if n>170: return
  if recover_box_max is None:
    recover_box_max=max(Integer(4000), Integer(800)*n)
  beta=2
  tbk=cputime()
  while beta<=n:
    print (beta)
    if beta==2:
      M5=ke.LLL()
      M5=M5[:n]  # this is for the affine case
    else:
#      M5=M5.BKZ(block_size=beta, strategies=BKZ.DEFAULT_STRATEGY, flags=BKZ.AUTO_ABORT|BKZ.GH_BND)
      M5=M5.BKZ(block_size=beta)
    
    # we succeed if all basis rows are in [-B,B] (same criterion as Step2_BKZ for bounded secrets)
    cl=len([True for v in M5 if allbounded(v,B)])
    if cl==n: break

    if beta==2:
      beta=10
    else:
      beta+=10  
  flag=0
  for v in M5:  
    if not allbounded(v,B): 
       flag=1
       print (v)
  
  print ("BKZ beta=%d: %.1f" % (beta,cputime(tbk))),
  if flag==1:
     MB=0
  else: 
      t2=cputime() 
      if B==1:
        MB=recoverBinary(M5,kappa)
      else:
        MB=recoverBox_v2(M5,B,max_size=recover_box_max)
      #print('MB:',MB)
      print ("  Recovery: %.1f" % cputime(t2)),
      print ("  Number of recovered vector=",MB.nrows()),
      nfound=len([True for MBi in MB if MBi in X.T])
      print ("  NFound=",nfound),  
      NS=MB.T
  # b=X*a=NS*ra
  #invNSn=matrix(Integers(x0),NS[:n]).inverse()
  #ra=invNSn*b[:n]
  #nrafound=len([True for rai in ra if rai in a])
  #print "  Coefs of a found=",nrafound,"out of",n,
  print ("  Total BKZ: %.1f" % cputime(tbk)),

  return MB,beta  
  
def orthoLattice_mat(H,x0):
 m,l=H.dimensions()
 M=identity_matrix(ZZ,m)
 
 M[:l,:l]=x0*M[:l,:l]
 H0i=Matrix(Integers(x0),H[:l,:l]).inverse()
 M[l:m,0:l]=-H[l:m,:]*H0i
 
 return M  
 