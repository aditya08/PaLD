function [C,U] = pald_triplet(D)

if D' ~= D 
    error('distance matrix must be symmetric');
end

n = size(D,1);
U = triu(2*ones(n),1); % init each conflict focus size to include 2 points
C = zeros(n);


% consider all unique triplets to compute conflict focus sizes
for x = 1:(n-1)
    for y = (x+1):n
        for z = (y+1):n
            if D(x,y) < D(x,z) && D(x,y) < D(y,z)     % xy is closest pair
                U(x,z) = U(x,z) + 1;
                U(y,z) = U(y,z) + 1;
            elseif D(x,z) < D(x,y) && D(x,z) < D(y,z) % xz is closest pair
                U(x,y) = U(x,y) + 1;
                U(y,z) = U(y,z) + 1;
            elseif D(y,z) < D(x, z) && D(y, z) < D(x, y) % yz is closest pair
                U(x,y) = U(x,y) + 1;
                U(x,z) = U(x,z) + 1;
            else % x,y,z are equidistant
                U(x,y) = U(x,y) + 1;
                U(x,z) = U(x,z) + 1;
                U(y,z) = U(y,z) + 1;
                
            end
        end
    end
end

% fill in lower triangle of U

U = U + U';

% initialize C with diagonal entries determined by row sums of 1/U
C = diag(sum(1./(U+diag(Inf*ones(n,1))),2)); 
% consider all unique triplets to compute contributions to cohesion
for x = 1:(n-1)
    for y = (x+1):n
        for z = (y+1):n
            if D(x,y) < D(x,z) && D(x,y) < D(y,z)     % xy is closest pair
                C(x,y) = C(x,y) + 1 / U(x,z);
                C(y,x) = C(y,x) + 1 / U(y,z);
            elseif D(x,z) < D(x,y) && D(x,z) < D(y,z) % xz is closest pair
                C(x,z) = C(x,z) + 1 / U(x,y);
                C(z,x) = C(z,x) + 1 / U(y,z);
            elseif D(y,z) < D(x, z) && D(y, z) < D(x, y) % yz is closest pair
                C(y,z) = C(y,z) + 1 / U(x,y);
                C(z,y) = C(z,y) + 1 / U(x,z);
            else % equidistant
                if D(x, y) == D(x, z)
                    C(x,y) = C(x,y) + 1 / U(x,z);
                    C(y,x) = C(y,x) + .5 / U(y,z);

                    C(x,z) = C(x,z) + 1 / U(x,y);
                    C(z,x) = C(z,x) + .5 / U(y,z);
                end
                if D(x, z) == D(y, z)
                    C(x,z) = C(x,z) + .5 / U(x,y);
                    C(z,x) = C(z,x) + 1 / U(y,z);

                    C(y,z) = C(y,z) + .5 / U(x,y);
                    C(z,y) = C(z,y) + 1 / U(x,z);
                end
                if D(x, y) == D(y, z)
                    C(x,y) = C(x,y) + .5 / U(x,z);
                    C(y,x) = C(y,x) + 1 / U(y,z);
                    C(y,z) = C(y,z) + 1 / U(x,y);
                    C(z,y) = C(z,y) + .5 / U(x,z);
                end

                fprintf("(x,y,z): (%d, %d, %d), ",x, y, z);
                fprintf("D(x,y):%.2f, D(x,z): %.2f, D(y,z): %.2f\n\n", D(x,y), D(x, z), D(y, z));

                
            end
        end
    end
end      
C = C/(n-1);
end