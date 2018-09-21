# Question from NKU group:
# given: 1/99^2 = 0.(a1, a2, a3, ..., an)
# what's the value of a1+a2+a3+...+an ?

# 1/9801
# = (\sum_{i=1, n} a_i * 10^{-i})*(1 + 10^{-n} + 10^{-2n} + ...)
# = (\sum_{i=1, n} a_i * 10^{-i})* 10^n / (10^n - 1)
# = (\sum_{i=0, n-1} a_i * 10^{i}) / (10^n - 1)

# so 9801=9*9*11*11 | 999999999...999
# that is 9*11*11 | 1111111111...111
# so we have 9*k 1's; if k is odd, then 111111111111...110 can be divided by 11, so k is even
# assume 18*k 1's : 11 | 1010101010101010...101
# 101010101010101010101 = 11*9182736455463728191 ( total 11 1's )
# so k is 11x, we have 9*2*11x = 198x, n should be 198 at least.

up = -1
down = 9801

# -7 // 3 gives -3, so make sure we have positive value.
if up*down < 0:
  sign = '-'
else:
  sign = ''
up = abs(up)
down = abs(down)


res = []

reminder = up % down
remLst = []

# we only check the reminder is not in the list
# if in list, then there is a loop
while not (reminder in remLst) :
  # we now checked this reminder
  remLst.append(reminder)
  # append the quotient to result
  res.append(str(reminder // down))
  # get the reminder
  reminder = reminder % down
  # move to next step
  reminder *= 10

print "sum of each loop digit is: " + str(sum([int(x) for x in res]))

print '{} / {} = '.format(up, down)
print sign + str(up / down) + '.(' + ''.join(res) + ')'

# Final result should be 0.(00010203040506......95969799) , no 98 in it.
