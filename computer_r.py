import gmpy2 as gm
import math
# for x in range(1,100):
#     y = ((426 * (2 ** 25)) / x) ** x
#     if y > 2**512:
#         print(x)

# for r in range(1,100):
#     a = gm.mul(426,2**25)
#     b = gm.div(a,r)
#     c = gm.mul(1,b**r)
#     d = gm.mpz(2**512)
#     if c>d:
#         print(r)



# for r in range(1,100):
#     a = math.ceil(426 / r)
#     b = gm.mul(426,2**25)
#     c = gm.div(b,r)
#     d = gm.mul(a,c**(2*r))
#     e = gm.mpz(2**3070)
#     if d>e:
#         print(r)

for r in range(1,100):
    t = math.ceil(1347 / r)
    b = gm.mul(t,2**25)
    c = gm.mul(t,b**(2*r))
    compare = gm.mpz(2**3070)
    if c>compare:
        print(r)

# print(y)
# y = (426 / x) * (( (426 * (2**25)) / x ) ** (2*x))