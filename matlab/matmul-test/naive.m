function C = naive(A,B)
% naive matmul with 2 nested loops

    C = zeros(size(A,1),size(B,2));
    for i = 1:size(C,1)
        for j = 1:size(C,2)
            C(i,j) = A(i,:)*B(:,j);
        end
    end

end