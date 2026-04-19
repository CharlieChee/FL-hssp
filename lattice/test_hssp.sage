import os, sys

# Self-locate so this script runs both from the repo root and from lattice/
_d = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()
if os.path.isfile(os.path.join(_d, "hssp.sage")):
    os.chdir(_d)
elif os.path.isfile("lattice/hssp.sage"):
    os.chdir("lattice")
load("hssp.sage")

n=80
H=hssp(n,-1)
H.gen_instance()
Lm=hssp_attack(H)
assert Lm[-2]==n
H.gen_instance(m=n*2)
Lnso=hssp_attack(H,'ns')
assert Lnso[-2]==n
H.gen_instance(m=n*4)
Lns=hssp_attack(H,'ns')
assert Lns[-2]==n
H.gen_instance(m=n^2)
Ls=hssp_attack(H,'statistical')
assert Ls[-2]==n

print("\n\n--------------------------->HSSP_n Test: success!\n")

kappa=n//2-2
H=hssp(n,kappa)
H.gen_instance()
Lm=hssp_attack(H)
assert Lm[-2]==n
Lns=hssp_attack(H,'ns')
assert Lns[-2]==n
print("\n\n--------------------------->HSSP_n^kappa Test: success!\n")
