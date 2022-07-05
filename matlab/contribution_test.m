clear;
rng(6);

b1 = 300;
time_matrix = ["matrix size","original method", "block method","optimal block method","triplet method","block triplet method"];
time_matrix = [time_matrix;"block size",b1,b1,b1,b1,b1];
for i = 1:10
% create a random distance matrix that is symmetric with diagonal elements
% equal to zeros
n = 500*i; 
d = rand(n);
%d = [0 1 2 3; 1 0 4 5; 2 4 0 6; 3 5 6 0];
D = (d+d')/2;
D = D - diag(diag(D));

%profile on

fprintf('orignal method, running on dimension %d\n', n)

tic
[C1,F] = pald_orig(D,1);
t1 = toc;


% b is different based on cache size, the machine cache size 
% in my computer is 9MB, which is around the size that could 
% store 1000 x 1000 matrix

% in the raw block method, the b should equal to sqrt(M)/3
% where M is the cache size


fprintf('block method, same dimension, using block size %d\n', b1)
tic
[C2,U] = pald_block(D,1,b1);
t2 = toc ;
fprintf('error in blocked method is %g\n', norm(C1-C2))

fprintf('optimal block method, same dimension, using block size %d\n', b1)
tic
[C3,U1] = pald_opt(D,1,b1);
t3 = toc; 
fprintf('error in optimal blocked method is %g\n', norm(C1-C3))



fprintf('triplet method, same dimension\n')
tic
[C4,U4] = pald_triplet(D);
t4 = toc ;
fprintf('error in triplet method is %g\n', norm(C1-C4))



fprintf('triplet block method, same dimension\n')
tic
[C5,U5] = pald_triplet_block(D,b1);
t5 = toc ;
fprintf('error in triplet method is %g\n', norm(C1-C5))

time_matrix = [time_matrix;n,t1,t2,t3,t4,t5];

end


writematrix(time_matrix,'Running_Time.xls','Sheet',1,'Range','A1:F6')
