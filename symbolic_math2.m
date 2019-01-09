clear all;
close all;

S12 = [1.0 8.0; 0.0 1.0];
S23 = [1.0 3.0; 0.0 1.0];
S34 = [1.0 8.0; 0.0 1.0];
M1 = [1.0 0.0; -0.25586592178771 1.0];
sym_f2 = sym('f2');
M2 = [1.0 0.0; -1/sym_f2 1.0];
sym_f3 = sym('f3');
M3 = [1.0 0.0; -1/sym_f3 1.0];
sym_f4 = sym('f4');
M4 = [1.0 0.0; -1/sym_f4 1.0];

II = [1.0 0.0; 0.0 1.0];
OO2 = II - M4*S34*M3*S23*M2*S12*M1

eq1 = OO2(1,1); 
eq2 = OO2(1,2);
eq3 = OO2(2,1);
eq4 = OO2(2,2);

% eqns = [eq1, eq2];
% vars = [sym_f2, sym_f3];
% S = solve(eqns, vars)
% vpa(S.f2)
% vpa(S.f3)

% eqns = [eq1, eq3];
% vars = [sym_f2, sym_f3];
% S = solve(eqns, vars)
% vpa(S.f2)
% vpa(S.f3)

% eqns = [eq2, eq3];
% vars = [sym_f2, sym_f3];
% S = solve(eqns, vars)
% vpa(S.f2)
% vpa(S.f3)

% eqns = [eq2, eq4];
% vars = [sym_f2, sym_f3];
% S = solve(eqns, vars)
% vpa(S.f2)
% vpa(S.f3)

% eqns = [eq3, eq4];
% vars = [sym_f2, sym_f3];
% S = solve(eqns, vars)
% vpa(S.f2)
% vpa(S.f3)

eqns = [eq1, eq2, eq3]; % eq4
vars = [sym_f2, sym_f3, sym_f4];
S = solve(eqns, vars)
vpa(S.f2)
vpa(S.f3)

eqns = [eq1, eq2, eq4]; % eq3
vars = [sym_f2, sym_f3, sym_f4];
S = solve(eqns, vars)
vpa(S.f2)
vpa(S.f3)

eqns = [eq1, eq3, eq4]; % eq2
vars = [sym_f2, sym_f3, sym_f4];
S = solve(eqns, vars)
vpa(S.f2)
vpa(S.f3)

eqns = [eq2, eq3, eq4]; % eq1
vars = [sym_f2, sym_f3, sym_f4];
S = solve(eqns, vars)
vpa(S.f2)
vpa(S.f3)





