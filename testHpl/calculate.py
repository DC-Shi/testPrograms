#!/usr/bin/env python2
# Front row: 0 1 4 6 8  9  12 13
# Back row:  2 3 5 7 10 11 14 15
# What we want:
# Give the number of current ranks $n \in \{1...16\}, param $a, b, c$
# the output $Y = \{y | y = ax^2+bx+c mod 16, 0 \le x \lt n \}$
# and make sure $Y$ is all in front row or back row.

front = [0, 1, 4, 6, 8, 9, 12, 13]
back = [2, 3, 5, 7, 10, 11, 14, 15]


def showMod(m=17):
  for a in range(0,m):
    print([pow(a,x)%m for x in range(0,16)])
 

MOD=17

def showN(n=8):
  print("============= Finding with n={} ".format(n))
  Y = []
  for f in range(0, 4):
   for e in range(0,16):
    for d in range(0,16):
     for c in range(0,16):
      for b in range(0,16):
       for a in range(0,16):
#        if all([a!=0, b!=0, c!=0, d!=0, e!=0]):
#          continue
        Y = [(pow(0,x)+a+b*x+c*pow(x,2)+d*pow(x,3)+e*pow(x,4)+f*pow(x,5))% MOD for x in range(n)]
        if len(list(set(Y))) != len(Y):
          continue # gpu number is not unique, abandon

        Y1 = [y in front for y in Y]
        if all(Y1):
          print("  Front row: Y = {}, a={}, b={}, c={}, d={}, e={}, f={}".format(Y, a, b, c, d, e, f))
          Y2 = [(pow(0,x)+a+b*x+c*pow(x,2)+d*pow(x,3)+e*pow(x,4)+f*pow(x,5))% MOD in back for x in range(8,16)]
          if all(Y2):
            print("  Bakc row: Y = {}, a={}, b={}, c={}, d={}, e={}".format(Y, a, b, c, d, e), f)
          continue

def print6(a, b, c, d, e, f):
  Y1 = [(pow(0,x)+a+b*x+c*pow(x,2)+d*pow(x,3)+e*pow(x,4)+f*pow(x,5))% MOD for x in range(8)]
  Y2 = [(pow(0,x)+a+b*x+c*pow(x,2)+d*pow(x,3)+e*pow(x,4)+f*pow(x,5))% MOD for x in range(8,16)]
  if all([y in front for y in Y1]):
    print("  Front row: Y1 = {}, a={}, b={}, c={}, d={}, e={}, f={}".format(Y1, a, b, c, d, e, f))
  else:
    print("not in front row")

  if all([y in back for y in Y2]):
    print("  Bakc row: Y2 = {}, a={}, b={}, c={}, d={}, e={}".format(Y2, a, b, c, d, e), f)
  else:
    print("not in back row")

# a=11, b=4, c=0, d=9, e=14, f=4
#0 0 1 0 1 0 1 1 0 0 1 1 0 0 0 0 0 0 a=12, b=5, c=0, d=15, e=5, f=6
#1 1 0 0 1 1 0 1 1 0 0 0 1 1 1 0 0 1 0 1 0 1 1 1 0 1 1 0 0 0 0 0 0 0 0 1 1 0 0 1 a=3, b=11, c=0, d=11, e=1, f=9
#0 1 0 0 0 1 0 1 0 0 0 0 a=7, b=6, c=0, d=4, e=4, f=9
#1 1 0 0 0 0 0 0 a=11, b=6, c=0, d=0, e=14, f=9

#print6(2,8,0,8,3,5)
#print6(10,5,2,11,3,9)
#print6(11,4,0,9,14,4)
#print6(12,5,0,15,5,6)
#print6(3,11,0,11,1,9)
#print6(7,6,0,4,4,9)
#print6(11,6,0,0,14,9)


def printCudaprintf():
  print( 'printf("a=%d, b=%d, c=%d, d=%d, e=%d, f=%d, Y=[')
  for i in range(20):
    print('%d'),
  print( ']\\n", a,b,c,d,e,f,')
  for i in range(20):
    print('Y[{}],'.format(i)),
  print(');')


printCudaprintf()

