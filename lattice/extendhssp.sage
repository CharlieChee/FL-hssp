

def genParams_mat(n=10,m=20,nx0=100,v=-1,B=1, l=10):
  assert v<=n, "v>n!"
  x0=genpseudoprime(nx0)


  # We generate the alpha_i's
  A=matrix(ZZ,n,l)
  for i in range(n):
    for j in range(l):
      A[i,j]=mod(ZZ.random_element(x0),x0)

  # The matrix X has m rows and must be of rank n
  if v>0:
    while True:
      X=Matrix(ZZ,m,n)
      for i in range(m):
        Lin=gen_list(n, v) 
        for j in Lin:
          X[i,j]=ZZ.random_element(1,B+1)
        del Lin
      #print (X.rank())
      if X.rank()==n: break
  else:
    while True:
      X=Matrix(ZZ,m,n)
      for i in range(m):
        for j in range(n):
          X[i,j]=ZZ.random_element(B+1)
      #print (X.rank())
      if X.rank()==n: break
  #print (X.density().n())

  # We generate an instance of the HSSP: b=X*A
  matB=X*A%x0


  return x0,A,X,matB
  
def Step1_Mat(n,x0,A,X,B,m):
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
  #print(MOrtho)
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

def Step2_BK_mat(ke,X,A,n,m,x0,B):
  print("Step2_BK_mat here")
  kappa=-1  
  #if n>170: return
  beta=2
  tbk=cputime()
  while beta<n:
    #print (beta)
    if beta==2:
      M5=ke.LLL()
      M5=M5[:n]  # this is for the affine case
    else:
#      M5=M5.BKZ(block_size=beta, strategies=BKZ.DEFAULT_STRATEGY, flags=BKZ.AUTO_ABORT|BKZ.GH_BND)
      M5=M5.BKZ(block_size=beta)
    
    # we succeed if we only get vectors with {-1,0,1} components, for kappa>0 we relax this condition to all except one vector
    #if len([True for v in M5 if allpmones(v)])==n: break
    cl=len([True for v in M5 if allbounded(v,1)])
    if cl==n: break
    #if kappa>0 and cl==n-1: break

    if beta==2:
      beta=10
    else:
      beta+=10  
  flag=0
  for v in M5:  
    if not allbounded(v,1): 
       flag=1
       #print (v)  #check if the reconstructed X is in the range of 1 -1 0, if not, then it fails
  
  print ("BKZ beta=%d: %.1f" % (beta,cputime(tbk))),
  if flag==1:
     MB=0
     print("error 0")
     return 0,0,0,0,0
  else: 
      t2=cputime() 
      MB=recoverBinary(M5,kappa)
      #print('MB:',MB)
      print ("  Recovery: %.1f" % cputime(t2)),
      print ("  Number of recovered vector=",MB.nrows()),
      nfound=len([True for MBi in MB if MBi in X.T])
      print ("  NFound=",nfound),  
  NS=MB.T
  ##B=X*A=NS*rA
  tmp = matrix(Integers(x0),NS[:n])
  if tmp.is_square():
    invNSn=tmp.inverse()
  else:
      print("error 1")
      return 1,1,1,1,1
  #rA=invNSn*B[:n]%x0
  rA=invNSn*B[:n]
  nrafound=len([True for rai in rA if rai in A])
  print ("  Coefs of a found=",nrafound,"out of",n),
  print ("  Total BKZ: %.1f" % cputime(tbk)),

  return MB,beta,rA,nfound, nrafound
  
def orthoLattice_mat(H,x0):

 #print(H)
 m,l=H.dimensions()
 M=identity_matrix(ZZ,m)

 #M[:l,:l]=x0*M[:l,:l]
 #H0i=Matrix(Integers(x0),H[:l,:l]).inverse() # invert the top-left l-block of H
 #M[l:m,0:l]=-H[l:m,:]*H0i

 M[:l,:l]=x0*M[:l,:l]
 H0i=Matrix(Integers(x0),H[:l,:l])

 #print(H0i)
 H0i=H0i.inverse() # invert the top-left l-block of H

 M[l:m,0:l]=-H[l:m,:]*H0i


 return M
  
  
  
  
  
  
  
  
