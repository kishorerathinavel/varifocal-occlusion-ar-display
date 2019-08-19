d_12 = 2
d_34 = 21
f_3 = 20

f_2 = (f_3 * d_12)/d_34

# f_2 = 10
# f_3 = (d_34*f_2)/(d_12)

print("f_3 = ", end = " ")
print(f_3)
d_23 = (d_34*(1 + f_2/f_3))/(d_34/f_3 - 1)
print("d_23 = ", end = " ")
print(d_23)

A = 1 - ((d_34 + d_23*(1 - d_34/f_3))/f_2) - (d_34/f_3)
# Should be 1
print("A should be 1. A = ", end = " ")
print(A)

C = d_23*(1 - d_34/f_3) + d_34 + d_12*A
print("C should be 0. C = ", end = " ")
print(C)
# Should be 0

vip_L1 = 6
slm_L1 = 2

f_1 = (vip_L1*slm_L1)/(vip_L1 + slm_L1)
print("f_1 = ", end = " ")
print(f_1)

B = ((1 - (d_23/f_3) - d_12*(((1 - d_23/f_3)/f_2) + (1/f_3)))/f_1) + ((1 - d_23/f_3)/f_2) + (1/f_3)
f_4 = -1/B
print("f_4 = ", end = " ")
print(f_4)



