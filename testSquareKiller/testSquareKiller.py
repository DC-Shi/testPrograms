# https://www.youtube.com/watch?v=myGqOF6blPI
# Square Killer: sum of digits inside a cage must be a square number.
# We only list possible ones.

def isSquare(n):
  if n in [4,9,16,25,36]:
    return True
  else:
    return False

def prints(array):
  strs = [str(a) for a in array]
  print("+".join(strs))

def sumPrint(array):
  if isSquare(sum(array)):
    prints(array)

def scanNdigit(digits):
  a = [0,0,0,0,0]
  for i in range(digits):
    a[i] = i+1


  while True:
    if a[0] > 9:
      break;

    #print(a)

    # Check from last digit, and give carry if needed.
    curIdx = digits - 1
    while a[curIdx] > (10-digits + curIdx):
      if curIdx == 0:
        break
      a[curIdx-1] += 1
      for idx in range(curIdx, digits):
        a[idx] = a[idx-1] + 1
      curIdx -= 1

    sumPrint(a)
    a[digits-1] += 1



for digits in range(2,6):
  print("================{}============".format(digits))
  scanNdigit(digits)
 
