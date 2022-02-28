import common_functions as cf

TC_bounds = [5, 12]
C_bounds = [10, 20]
TC_bounds_diopters = [cf.convert_cm2dpt(elem) for elem in TC_bounds]
C_bounds_diopters = [cf.convert_cm2dpt(elem) for elem in C_bounds]
print(TC_bounds)
print(TC_bounds_diopters)
print(C_bounds)
print(C_bounds_diopters)

offset_lens = -15
nC_bounds = [cf.convert_dpt2cm(cf.convert_cm2dpt(offset_lens) + cf.convert_cm2dpt(elem)) for elem in C_bounds]
print(nC_bounds)

res_bounds = [-66.6, 28.6]
nC_bounds = [cf.convert_dpt2cm(cf.convert_cm2dpt(elem) - cf.convert_cm2dpt(offset_lens)) for elem in C_bounds]
print(nC_bounds)
