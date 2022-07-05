function [naivetime,blockedtime,opt_blockedtime] = test(n,b)

    A = randn(n);
    B = randn(n);
    
    tic, C = naive(A,B); naivetime = toc;
    tic, D = blocked(A,B,b); blockedtime = toc;
    tic, E = optimal_blocked(A,B,b); opt_blockedtime = toc;
    
    assert(norm(C-D)/norm(C) < 1e-14);
    assert(norm(C-E)/norm(C) < 1e-14);

end