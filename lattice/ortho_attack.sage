# Computes the right kernel of M using LLL.
# We assume that m>=2*n. This is only to take K proportional to M.height()
# We follow the approach from https://hal.archives-ouvertes.fr/hal-01921335/document
from time import time
def kernelLLL(M):
  n=M.nrows()
  m=M.ncols()
  if m<2*n: return M.right_kernel().matrix()
  K=2^(m//2)*M.height()
  
  MB=Matrix(ZZ,m+n,m)
  # remember to revert
  # MB = Matrix(QQ, m + n, m)
  MB[:n]=K*M
  MB[n:]=identity_matrix(m)
  
  MB2=MB.T.LLL().T

  assert MB2[:n,:m-n]==0
  # remember to revert
  Ke=MB2[n:,:m-n].T

  return Ke
  

def matNbits(M):
  return max([M[i,j].nbits() for i in range(M.nrows()) for j in range(M.ncols())])
# remember to revert
#
# def matNbits(M):
#   max_bits = 0
#   for i in range(M.nrows()):
#     for j in range(M.ncols()):
#       element = M[i, j]
#       if element.parent() == QQ:  # if it is a rational number
#         numerator_bits = element.numerator().nbits()
#         denominator_bits = element.denominator().nbits()
#         max_bits = max(max_bits, max(numerator_bits, denominator_bits))
#       else:
#         max_bits = max(max_bits, element.nbits())
#   return max_bits

def NZeroVectors(M):
  return sum([vi==0 and 1 or 0 for vi in M])


# Matrix rounding to integers
def roundM(M):
  M2=Matrix(ZZ,M.nrows(),M.ncols())
  for i in range(M.nrows()):
    for j in range(M.ncols()):
      M2[i,j]=round(M[i,j])
  return M2
  


def Step1_plus_gen(n,v,m=0,BKZ=False):     
  if m==0 and n % 2==1:
    m=n*(n+3)/2 # n is odd
  elif m==0 and n % 2 ==0:
    m=n*(n+4)/2 # n is even
  k=4

  print( "n=",n,"m=",m,"k=",k),
  if v>-1: print ("kappa=",v),

  iota=0.035
  nx0=int(2*iota*n^2+n*log(n,2))
  print ("nx0=",nx0)

  x0,a,X,b=genParams(n,m,nx0,v)

  M=orthoLatticeMod(b,n,x0)

  print( "Step 1"),
  t=cputime()

  M[:n//k,:n//k]=M[:n//k,:n//k].LLL()
  
  M2=M[:2*n,:2*n].LLL()
  tprecomp=cputime(t)
  print ("  LLL:%.1f" % tprecomp),

  RF=RealField(matNbits(M))

  M4i=Matrix(RF,M[:n//k,:n//k]).inverse()
  M2i=Matrix(RDF,M2).inverse()  
    
  ts1=cputime()
  while True:
    flag=True
    for i in range((Integer(m/n)-2)*k):
      indf=2*n+n//k*(i+1)
      if i==(Integer(m/n)-2)*k-1:
        indf=m
        
      mv=roundM(M[2*n+n//k*i:indf,:n//k]*M4i)
      if mv==0: 
        continue
      flag=False
      M[2*n+n//k*i:indf,:]-=mv*M[:n//k,:]
    if flag: break
  print( "  Sred1:%.1f" % cputime(ts1)),

  M[:2*n,:2*n]=M2

  ts2=cputime()
  while True:
    #print "  matNBits(M)=",matNbits(M[2*n:])
    mv=roundM(M[2*n:,:2*n]*M2i)
    if mv==0: break
    M[2*n:,:]-=mv*M[:2*n,:]
  tt10=cputime(t)
  print ("  Sred2:%.1f" % cputime(ts2)),
  
  # The first n vectors of M should be orthogonal
  northo=NZeroVectors(M[:n,:2*n]*X[:2*n])

  for i in range(2,Integer(m/n)):
    northo+=NZeroVectors(M[i*n:(i+1)*n,:2*n]*X[:2*n]+X[i*n:(i+1)*n])

  print ("  #ortho vecs=",northo,"out of",m-n),
  
  # Orthogonal of the orthogonal vectors
  # We compute modulo 3 if multivariate
  if BKZ: KK=ZZ
  else: KK=GF(3)
  MO=Matrix(KK,n,m)
  
  tk=cputime()
  MO[:,:2*n]=kernelLLL(M[:n,:2*n])
  print ("  Kernel LLL: %.1f" % cputime(tk)),

  for i in range(2,Integer(m/n)):
    MO[:,i*n:(i+1)*n]=-(M[i*n:(i+1)*n,:2*n]*MO[:,:2*n].T).T
  #print "Total kernel computation",cputime(tk)
  tt1=cputime(t)
  tt1O=cputime(tk)
  print ("  Total Step 1: %.1f" % tt1)
  #L=IntegerLattice(MO)
  #Y=X.T
  #R=Y.rows()
  #print all(w in L for w in R)
  #LX=IntegerLattice(Y)
  #print all(w in LX for w in MO.rows())
  
  del M,mv
  if BKZ:return MO,x0,a,X,b,int(m),tt1,tt10,tt1O
  else: return MO,x0,a,X,b,int(m),tt1
  

def Step1(n,v,x0,a,X,b,m,BKZ=False):   
  k=4
  if n>200: k=5
  print ("Building M "),
  #M=orthoLatticeMod(b,n,x0)

  M = orthoLattice_mat(b, x0)
  print ("Step 1")
  t=cputime()


  M = Matrix(ZZ, M)

  M[:n//k,:n//k]=M[:n//k,:n//k].LLL()


  M2=M[:2*n,:2*n].LLL()

  tprecomp=cputime(t)
  print ("  LLL:%.1f" % tprecomp),

  RF=RealField(matNbits(M))


  M4i=Matrix(RF,M[:n//k,:n//k]).inverse()
  M2i=Matrix(RDF,M2).inverse()

  ts1=cputime()
  start_time = time()

  while True:
    flag = True  # Initialize the flag to track if updates are made in this iteration

    # Iterate over blocks of the matrix, divided based on (m / n - 2) * k
    for i in range((int(m / n) - 2) * k):
      # Compute the end index for the current block
      indf = 2 * n + n // k * (i + 1)


      # If this is the last block, ensure indf does not exceed m
      if i == (int(m / n) - 2) * k - 1:
        indf = m

      # Perform the Babai rounding step on the current block of the matrix
      # `roundM` approximates the projection of the matrix block onto the space defined by M4i
      mv = roundM(M[2 * n + n // k * i:indf, :n // k] * M4i)

      # If `mv` (resulting vector from rounding) is zero, skip this block
      if mv == 0:
        continue

      # Update the flag since a non-zero update is made
      flag = False

      # Adjust the current block of the matrix by subtracting the scaled portion of the base matrix
      M[2 * n + n // k * i:indf, :] -= mv * M[:n // k, :]

    # Check if the process exceeds 20 seconds; if so, terminate to avoid excessive runtime
    if time() - start_time > 3:
      print("Execution time exceeded 20 seconds, terminating loop")
      return -1, 0, 0, 0

    # If no updates were made in this iteration, exit the loop
    if flag:
      break

  print ("  Sred1:%.1f" % cputime(ts1)),



  M[:2*n,:2*n]=M2




  ts2=cputime()
  start_time = time()
  while True:
    #print "  matNBits(M)=",matNbits(M[2*n:])
    mv=roundM(M[2*n:,:2*n]*M2i)
    if mv==0: break
    M[2*n:,:]-=mv*M[:2*n,:]
    if time() - start_time > 3:
      print("execution time exceeded 20 seconds, terminating loop")
      return -1,0,0,0
  tsr=cputime(ts2)
  tt10=cputime(t)
  print ("  Sred2:%.1f" % cputime(ts2)),
  print("step 1 here")
  # The first n vectors of M should be orthogonal


  northo=NZeroVectors(M[:n,:2*n]*X[:2*n])

  for i in range(2,int(m/n)):
    northo+=NZeroVectors(M[i*n:(i+1)*n,:2*n]*X[:2*n]+X[i*n:(i+1)*n])

  print( "  #ortho vecs=",northo,"out of",m-n),
  print('stop')

  # Orthogonal of the orthogonal vectors
  # We compute modulo 3 if multivariate
  if BKZ: KK=ZZ
  else: KK=GF(3)
  MO=Matrix(KK,n,m)
  # MO = Matrix(QQ, n, m)
  # remember to revert
  tk=cputime()
  MO[:,:2*n]=kernelLLL(M[:n,:2*n])
  print ("  Kernel LLL: %.1f" % cputime(tk)),



  for i in range(2,int(m/n)):
    MO[:,i*n:(i+1)*n]=-(M[i*n:(i+1)*n,:2*n]*MO[:,:2*n].T).T

  #print "Total kernel computation",cputime(tk)
  tt1=cputime(t)
  tt1O=cputime(tk)
  print( "  Total Step 1: %.1f" % tt1)
  #L=IntegerLattice(MO)
  #Y=X.T
  #R=Y.rows()
  #print all(w in L for w in R)
  #LX=IntegerLattice(Y)
  #print all(w in LX for w in MO.rows())
  return MO,tt1,tt10,tt1O
  #else: return MO, tt1
  


# We generate the lattice of vectors orthogonal to b modulo x0 
# and also to c in the affine case
def orthoLattice(b,x0):
 m=b.length()
 #print("m",m)
 M=Matrix(ZZ,m,m)

 for i in range(1,m):
      M[i,i]=1
      

 M[1:m,0]=-b[1:m]*inverse_mod(b[0],x0)  #Jane: each row vector is orthogonal to b
 M[0,0]=x0

 for i in range(1,m):
      M[i,0]=mod(M[i,0],x0)

 return M
  

def Step1_original(n,v,x0,a,X,b,m):
  #M=orthoLattice(b,x0)
  print("calculate M ......")

  M = orthoLattice_mat(b, x0)

  print ("Step 1"),
  t=cputime()
  M2=M.LLL()
  #print(M2)
  tt10=cputime(t)
  print ("LLL step1: %.1f" % cputime(t)),

  #assert sum([vi==0 and 1 or 0 for vi in M2*X])==m-n

  MOrtho=M2[:m-n]

  #print
  #for i in range(m-n+1):
    #print (i,N(log(M2[i:i+1].norm(),2)),N(log(m^(n/(2*(m-n)))*sqrt((m-n)/17),2)+iota*m+nx0/(m-n))) #N(log(sqrt((m-n)*n)*(m/2)^(m/(2*(m-n))),2)+iota*m)

  #print ("  log(Height,2)=",int(log(MOrtho.height(),2))),

  t2=cputime()
  #print(MOrtho)
  print(" - ")
  ke=kernelLLL(MOrtho)
  print(ke)
  tt1O=cputime(t2)
  print( "  Kernel: %.1f" % cputime(t2)),
  tt1=cputime(t)
  print ("  Total step1: %.1f" % tt1)

  return ke,tt1,tt10,tt1O


def Step1_Mat(n,v,x0,a,X,b,m,BKZ=False):   
  k=4
  if n>200: k=5
  print ("Building M "),
  M=orthoLatticeMod(b,n,x0)
  print ("Step 1"),
  t=cputime()
  M[:n//k,:n//k]=M[:n//k,:n//k].LLL()
  
  M2=M[:2*n,:2*n].LLL()
  tprecomp=cputime(t)
  print ("  LLL:%.1f" % tprecomp),

  RF=RealField(matNbits(M))

  M4i=Matrix(RF,M[:n//k,:n//k]).inverse()
  M2i=Matrix(RDF,M2).inverse()  
    
  ts1=cputime()
  while True:
    flag=True
    for i in range((Integer(m/n)-2)*k):
      indf=2*n+n//k*(i+1)
      if i==(Integer(m/n)-2)*k-1:
        indf=m
        
      mv=roundM(M[2*n+n//k*i:indf,:n//k]*M4i)   #Jane: Babai
      if mv==0: 
        continue
      flag=False
      M[2*n+n//k*i:indf,:]-=mv*M[:n//k,:]
    if flag: break
  print ("  Sred1:%.1f" % cputime(ts1)),

  M[:2*n,:2*n]=M2

  ts2=cputime()
  while True:
    #print "  matNBits(M)=",matNbits(M[2*n:])
    mv=roundM(M[2*n:,:2*n]*M2i)
    if mv==0: break
    M[2*n:,:]-=mv*M[:2*n,:]
  tsr=cputime(ts2)
  tt10=cputime(t)
  print ("  Sred2:%.1f" % cputime(ts2)),
  
  # The first n vectors of M should be orthogonal
  northo=NZeroVectors(M[:n,:2*n]*X[:2*n])

  for i in range(2,Integer(m/n)):
    northo+=NZeroVectors(M[i*n:(i+1)*n,:2*n]*X[:2*n]+X[i*n:(i+1)*n])

  print( "  #ortho vecs=",northo,"out of",m-n),
  
  # Orthogonal of the orthogonal vectors
  # We compute modulo 3 if multivariate
  if BKZ: KK=ZZ
  else: KK=GF(3)
  MO=Matrix(KK,n,m)

  tk=cputime()
  MO[:,:2*n]=kernelLLL(M[:n,:2*n])
  print ("  Kernel LLL: %.1f" % cputime(tk)),

  for i in range(2,Integer(m/n)):
    MO[:,i*n:(i+1)*n]=-(M[i*n:(i+1)*n,:2*n]*MO[:,:2*n].T).T
  #print "Total kernel computation",cputime(tk)
  tt1=cputime(t)
  tt1O=cputime(tk)
  print( "  Total Step 1: %.1f" % tt1)
  #L=IntegerLattice(MO)
  #Y=X.T
  #R=Y.rows()
  #print all(w in L for w in R)
  #LX=IntegerLattice(Y)
  #print all(w in LX for w in MO.rows())
  
  del M,mv
  return MO,tt1,tt10,tt1O
  #else: return MO, tt1



def orthoLattice_mat(H,x0):

 if isinstance(H, sage.modules.vector_integer_dense.Vector_integer_dense):
     return orthoLattice_single(H,x0)
 m,l=H.dimensions()
 M=identity_matrix(ZZ,m)
 # remember to revert
 # M = identity_matrix(QQ, m)
 M[:l,:l]=x0*M[:l,:l]
 H0i=Matrix(Integers(x0),H[:l,:l])
 # remember to revert
 # H0i = Matrix(QQ, [[(x - x0 * int(x // x0)) for x in row] for row in H[:l, :l].rows()])
 # H0i = Matrix(RR, [[(x - x0 * floor(x / x0)) for x in row] for row in H[:l, :l].rows()])
 print(H0i)
 H0i=H0i.inverse()

 M[l:m,0:l]=-H[l:m,:]*H0i

 #print(M)

 return M

def orthoLattice_single(b,x0):
 m=b.length()
 M=Matrix(ZZ,m,m)

 for i in range(1,m):
      M[i,i]=1
 M[1:m,0]=-b[1:m]*inverse_mod(b[0],x0)
 M[0,0]=x0

 for i in range(1,m):
      M[i,0]=mod(M[i,0],x0)

 return M

