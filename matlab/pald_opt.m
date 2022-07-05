function [C,U] = pald_opt(D,beta,b)
% run optimial block algorithm on PaLD method
% b <= sqrt(M+1)-1, where M is the cache size


% error checking
if beta < 0
    error('beta must be positive');
end

if ~issymmetric(D) 
    error('distance matrix must be symmetric');
end

n = size(D,1);
C = zeros(n);
U = zeros(n);

% loop over blocks to update cohesion matrix
for i = 1:b:n
    Ib = i:min(i+b-1,n);
    for j = i:b:n
        Jb = j:min(j+b-1,n);
        
        % set temporary buffers
        Uij = zeros(min(b,n-i+1),min(b,n-j+1)); % conflict focus sizes
        Dij = D(Ib,Jb); % distances
        
        % compute size of conflict focus for block I,J
        for k=1:n
            % outer-product-like update
            Uij = Uij + double(min(D(Ib,k),D(Jb,k)') <= Dij);
        end       
        
        
        % update cohesion values according to conflict foci (I,J)
        % for each other point k (outer-product like updates)    
        if i == j % handle diagonal case differently
            
            % fix conflict focus of point with itself to be infinite
            Uij = Uij + diag(Inf*ones(min(b,n-i+1),1));
            
            % loop over contribution points
            for k=1:n
                % determine contributions to C(Ib,k) (ignore Jb because Jb == Ib)
                C(Ib,k) = C(Ib,k) + sum(double((D(Ib,k) < D(Jb,k)') & (D(Ib,k) <= Dij)) ./ Uij,2);
                C(Ib,k) = C(Ib,k) + .5*sum(double((D(Ib,k) == D(Jb,k)') & (D(Ib,k) <= Dij)) ./ Uij,2);            
            end
            
        else % general case         
            
            % loop over contribution points
            for k=1:n
                % determine contributions to C(Ib,k)
                C(Ib,k) = C(Ib,k) + sum(double((D(Ib,k) < D(Jb,k)') & (D(Ib,k) <= Dij)) ./ Uij,2);
                C(Ib,k) = C(Ib,k) + .5*sum(double((D(Ib,k) == D(Jb,k)') & (D(Ib,k) <= Dij)) ./ Uij,2);            
                % determine contributions to C(Jb,k)
                C(Jb,k) = C(Jb,k) + sum(double((D(Ib,k) > D(Jb,k)') & (D(Jb,k)' <= Dij)) ./ Uij)';
                C(Jb,k) = C(Jb,k) + .5*sum(double((D(Ib,k) == D(Jb,k)') & (D(Jb,k)' <= Dij)) ./ Uij)';
            end
        end
        
        U(Ib,Jb) = Uij; % copy to output for debugging
    end
end


C = C/(n-1);


end
