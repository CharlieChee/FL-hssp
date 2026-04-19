import numpy as np
#from fica import ICA

def ICA(X, B=1, n=16, kappa=-1, exn=0):
  t = time()
  X = np.asarray(X)
  ncomp, sample = X.shape
  # print(X.shape)
  ica = FastICA(n_components=ncomp, fun='cube')
  S_ = ica.fit_transform(X.T)
  A_ = ica.mixing_
  # assert np.allclose(X.T, np.dot(S_, A_.T) + ica.mean_)

  # we want to remove the mean
  # print(A_)
  # print(np.linalg.inv(A_.T).shape)
  s0 = np.dot(ica.mean_, np.linalg.inv(A_.T))
  S_ += s0

  # assert np.allclose(X, np.dot(A_,S_.T))  # now the mean is 0

  # we want that the components of S_ are positive
  # if the average of a column of S_ is negative, we take the opposite
  Sm = np.sign(np.average(S_, axis=0))
  S_ *= Sm
  A_ *= Sm

  # assert np.allclose(X, np.dot(A_,S_.T))

  # we want that the components of S_ are 0 or 1
  # If all the components of S_ are 0 or 1, the standard deviation will be 1/2
  # If the components are uniformly distributed between 0 and B, the standard deviation is
  # sqrt(B(B+2)/12)
  # So we divide the components of S_ by std/sqrt(B(B+2)/12), where std is computed column by column

  if kappa == -1 and exn == 0:
    exn = math.sqrt(B * (B + 2) / 12.)
  elif exn == 0:
    exn = math.sqrt((2 * B + 1) * (B + 1) * kappa / (n * 6.) - (
              kappa * (B + 1) / (2 * n)) ** 2)  # exn=math.sqrt((2*B+1)*(B+1)*kappa/(n*6.))

  st = S_.std(axis=0) / exn
  S_ /= st
  A_ *= st

  # print " ICA.py %.2f" %(time()-t),
  # assert np.allclose(X, np.dot(A_,S_.T))
  np.save('./ICA/A_.npy', A_)

  np.save('./ICA/S.npy', S_.T)

  return A_, S_.T


def Sage2np(MO,n,m):
  MOn=np.matrix(MO)
  return MOn

def runICA(MOn,B=1):
  t1=cputime()
  A_,S=ICA(MOn,B)
  A_=np.load('./ICA/A_.npy')
  S=np.load('./ICA/S.npy')
  #print('S=',S)
  #print('A=',A_)
  S_=np.dot(np.linalg.inv(A_),MOn)
  print ("time Ica_S: ", cputime(t1)),
  print('S2 before rounding',S2)
  S2=matrix(ZZ,MOn.shape[0],MOn.shape[1], round(S_)) 
  return S2

def runICA_A(MOn,B=1,n=16,kappa=-1):
  t1=cputime()
  A_,S=ICA(MOn,B,n,kappa)

  print("ica here")
  # A_=np.load('./ICA/A_.npy')
  #S=np.load('./ICA/S.npy')
  #print('S=',S)
  #print('A=',A_)
  print ("time Ica_A: ", cputime(t1)),
  A2=matrix(ZZ,MOn.shape[0],MOn.shape[0],round(A_))

  return A2


def statistical_1(MO,n,m,x0,X,a,b,kappa,B=1,variant=None):  
  
  if variant==None:
    if n<=200: variant='roundA'
    else: variant='roundX'

  print ("Step 2-ICA: ", variant)
  
  
  #print "matNbits=",matNbits(MO),
  tlll=cputime()
  # intefere
  # MO[0, 10] += 1
  #print('MO before LLL',MO)
  MO=MO.LLL()
  print (" time LLL=",cputime(tlll),"mathNbits=",matNbits(MO)),
  print("global here --------------------------")

  # intefere
  # MO[0, 10] += 7

  MOn=Sage2np(MO,n,m)
  #print('MO after LLL',MO)
  #print('MOn',MOn)
  np.save('./ICA/MOn.npy',MOn)
  #print("type MOn")
  #print(type(MOn))
  return MOn,MO

def statistical_2(MOn,MO,n,m,x0,X,a,b,kappa,B=1,variant=None):  
  t2=cputime()
  if variant==None:
    if n<=200: variant='roundA'
    else: variant='roundX'
  if variant=="roundA":
    A2=runICA_A(MOn,B,n,kappa)
    try:
      S2 = A2.inverse() * MO
      print("mathNbits A=", matNbits(A2)),
    except:
      return 0,0,0,0
  elif variant=="roundX":
    S2=runICA(MOn,B)
    print ("mathNbits X=",matNbits(X)),
  else:
    raise NameError('Variant algorithm non acceptable') 
  tica=cputime(t2)
  print( " cputime ICA %.2f" %tica),  

  tc=cputime()
  Y=X.T
  #print(Y)
  nfound=0
  for i in range(n):
    for j in range(n):
      if S2[i,:n]==Y[j,:n] and S2[i]==Y[j]:  
        nfound+=1
  t=cputime(tc)      
  print( "  NFound=",nfound,"out of",n,"check= %.2f" %t)

#  print('S2=',S2)
#  print('Y=',Y)
#  if nfound<n: 
#     return tica, 0 ,nfound

  NS=S2.T
  resX=np.save('./ICA/resX.npy',S2)

  
  tcoff=cputime()
  #b=X*a=NS*ra
  invNSn=matrix(Integers(x0),NS[:n]).inverse()
  ra=invNSn*b[:n]
  #print('Real a=',a)
  #print('Reconstructed a=',ra)
  tcf= cputime(tcoff)

  nrafound=len([True for rai in ra if rai in a])   
  print( "  Coefs of a found=",nrafound,"out of",n, " time= %.2f" %tcf)
  
  tS2=tcf+tica
  print( "  Total step2: %.1f" % tS2),
  
  return tica, tS2, nrafound, nfound


def statistical_2_mat(MOn, MO, n, m, x0, X, a, b, kappa, B=1, variant=None):
  print("statistical_2_mat here")
  t2 = cputime()
  if variant == None:
    if n <= 200:
      variant = 'roundA'
    else:
      variant = 'roundX'
  if variant == "roundA":
    A2 = runICA_A(MOn, B, n, kappa)
    # print(B)
    # print(A2)
    try:
      S2 = A2.inverse() * MO
      # print(S2)
      print("mathNbits A=", matNbits(A2)),
    except:
      print("wrong here")
      return 0, 0, 0,0,0,0
  elif variant == "roundX":
    S2 = runICA(MOn, B)
    print("mathNbits X=", matNbits(X)),
  else:
    raise NameError('Variant algorithm non acceptable')
  tica = cputime(t2)
  print(" cputime ICA %.2f" % tica),

  tc = cputime()
  Y = X.T
  # print(Y)
  nfound = 0
  for i in range(n):
    for j in range(n):
      if S2[i, :n] == Y[j, :n] and S2[i] == Y[j]:
        nfound += 1
  t = cputime(tc)
  print("  NFound=", nfound, "out of", n, "check= %.2f" % t)

  #  print('S2=',S2)
  #  print('Y=',Y)
  #  if nfound<n:
  #     return tica, 0 ,nfound

  NS = S2.T
  resX = np.save('./ICA/resX.npy', S2)

  tcoff = cputime()
  # b=X*a=NS*ra
  # invNSn = matrix(Integers(x0), NS[:n]).inverse()
  invNSn = matrix(QQ, NS[:n]).inverse()
  # remember to revert
  ra = invNSn * b[:n]
  # print('Real a=',a)
  # print('Reconstructed a=',ra)
  tcf = cputime(tcoff)

  nrafound = len([True for rai in ra if rai in a])
  print("  Coefs of a found=", nrafound, "out of", n, " time= %.2f" % tcf)

  tS2 = tcf + tica
  print("  Total step2: %.1f" % tS2),

  return tica, tS2, nrafound, NS.T,nfound,nrafound
