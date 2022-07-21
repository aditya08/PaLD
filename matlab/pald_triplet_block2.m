function [C,U] = pald_triplet_block2(D,b)


if D' ~= D 
    error('distance matrix must be symmetric');
end

n = size(D,1);
U = triu(2*ones(n),1); % init each conflict focus size to include 2 points
C = zeros(n);



% consider all unique triplet blocks to compute conflict focus sizes
for x = 1:b:n
    Xb = x:min(x+b-1,n);
    for y = x:b:n
        Yb = y:min(y+b-1,n);
        for z = y:b:n
            Zb = z:min(z+b-1,n);
            [UXY,UXZ,UYZ] = ...
            update_cfs(x, y, z, D(Xb,Yb), D(Xb,Zb), D(Yb,Zb));
            if x == y && y == z
%                 U(Xb, Yb) = UXY + UXZ + UYZ;
                U(Xb, Yb) = U(Xb, Yb) + UXY + UXZ + UYZ;
%                 Xb
%                 Yb
%                 UXY
%                 [U(Xb,Yb)] = update_cfs(U(Xb,Yb),D(Xb,Yb));
            elseif x == y
                U(Xb,Yb) = U(Xb,Yb) + UXY;
                U(Xb,Zb) = U(Xb,Zb) + UXZ + UYZ;
%                 UXY
%                 UXZ
%                 [U(Xb,Yb),U(Xb,Zb)] = update_cfs1(U(Xb,Yb),U(Xb,Zb),D(Xb,Yb),D(Xb,Zb));
            elseif y == z
                U(Xb,Zb) = U(Xb,Zb) + UXY + UXZ;
                U(Yb,Zb) = U(Yb,Zb) + UYZ;
%                 UXZ
%                 UYZ
%                 [U(Xb,Zb),U(Yb,Zb)] = update_cfs2(U(Xb,Zb),U(Yb,Zb),D(Xb,Zb),D(Yb,Zb));
            else
                U(Xb,Yb) = U(Xb,Yb) + UXY;
                U(Xb,Zb) = U(Xb,Zb) + UXZ;
                U(Yb,Zb) = U(Yb,Zb) + UYZ;
%                 [U(Xb,Yb),U(Xb,Zb),U(Yb,Zb)] = ...
%             update_cfs3(U(Xb,Yb),U(Xb,Zb),U(Yb,Zb),D(Xb,Yb),D(Xb,Zb),D(Yb,Zb));
            end
            
        end
    end
end


% fill in lower triangle of U
 U = U + U';

% initialize C with diagonal entries determined by row sums of 1/U
C = diag(sum(1./(U+diag(Inf*ones(n,1))),2)); 




% consider all unique triplets to compute contributions to cohesion


for x = 1:b:n
    Xb = x:min(x+b-1,n);
    for y = x:b:n
        Yb = y:min(y+b-1,n);
        for z = y:b:n
            Zb = z:min(z+b-1,n);
            [CXY,CXZ,CYZ,CYX,CZX,CZY] = ...
            update_coh3(x, y, z, U(Xb,Yb),U(Xb,Zb),U(Yb,Zb),D(Xb,Yb),D(Xb,Zb),D(Yb,Zb));
            if x == y && y == z
                C(Xb,Yb) = C(Xb,Yb) + CXY + CYX + CXZ + CZX + CYZ + CZY;
            elseif x == y
                C(Xb,Yb) = C(Xb,Yb) + CXY + CYX;
                C(Xb,Zb) = C(Xb,Zb) + CXZ + CYZ;
                C(Zb,Xb) = C(Zb,Xb) + CZX + CZY;
            elseif y == z
                C(Xb,Zb) = C(Xb,Zb) + CXZ + CXY;
                C(Yb,Zb) = C(Yb,Zb) + CYZ + CZY;
                C(Zb,Xb) = C(Zb,Xb) + CZX + CYX;
            else
                C(Xb,Yb) = C(Xb,Yb) + CXY;
                C(Xb,Zb) = C(Xb,Zb) + CXZ;
                C(Yb,Zb) = C(Yb,Zb) + CYZ;
                C(Yb,Xb) = C(Yb,Xb) + CYX;
                C(Zb,Xb) = C(Zb,Xb) + CZX;
                C(Zb,Yb) = C(Zb,Yb) + CZY;
            end
            C/(n-1);
        end
    end
end    

C = C/(n-1);
end

function [Uxy,Uxz,Uyz] = update_cfs(xb, yb, zb, Dxy, Dxz, Dyz)
% update conflict focus sizes for triplets.

    [m,n] = size(Dxy);
    [~,p] = size(Dxz);
 
    Uxy = zeros(m,n);
    Uxz = zeros(m,p);
    Uyz = zeros(n,p);

    xend = m;
    ystart = 1;
    zstart = 1;
    if (xb == yb) && (yb == zb)
        xend = m - 1;
    end
    for x = 1:xend
        if xb == yb
            ystart = x + 1;
        end
        for y=ystart:n
            if yb == zb
                zstart = y + 1;
            end
            for z=zstart:p
                if Dxy(x,y) < Dxz(x,z) && Dxy(x,y) < Dyz(y,z)       % xy is closest pair
                    Uxz(x,z) = Uxz(x,z) + 1;
                    Uyz(y,z) = Uyz(y,z) + 1;
                elseif Dxz(x,z) < Dxy(x,y) && Dxz(x,z) < Dyz(y,z)   % xz is closest pair
                    Uxy(x,y) = Uxy(x,y) + 1;
                    Uyz(y,z) = Uyz(y,z) + 1;
                else                                                % yz is closest pair
                    Uxy(x,y) = Uxy(x,y) + 1;
                    Uxz(x,z) = Uxz(x,z) + 1;
                end
            end
        end
    end
end


function [Cxy] = update_coh(Uxy,Dxy,Cxy)
% update Cohension matrix block when x == y == z

    [n,~] = size(Dxy);
    
    for x = 1:(n-1)
        for y = (x+1):n
            for z = (y+1):n
                if Dxy(x,y) < Dxy(x,z) && Dxy(x,y) < Dxy(y,z)     % xy is closest pair
                    Cxy(x,y) = Cxy(x,y) + 1 / Uxy(x,z);
                    Cxy(y,x) = Cxy(y,x) + 1 / Uxy(y,z);
                elseif Dxy(x,z) < Dxy(x,y) && Dxy(x,z) < Dxy(y,z) % xz is closest pair
                    Cxy(x,z) = Cxy(x,z) + 1 / Uxy(x,y);
                    Cxy(z,x) = Cxy(z,x) + 1 / Uxy(y,z);
                else                                      % yz is closest pair
                    Cxy(y,z) = Cxy(y,z) + 1 / Uxy(x,y);
                    Cxy(z,y) = Cxy(z,y) + 1 / Uxy(x,z);
                end
            end
        end
    end    
    
end

function [Cxy,Cxz,Czx] = update_coh1(Uxy,Uxz,Dxy,Dxz,Cxy,Cxz,Czx)
% update Cohension matrix block when x == y 

    [m,n] = size(Dxz);
    
    for x = 1:m
        for y=(x+1):m
            for z=1:n

                if Dxy(x,y) < Dxz(x,z) && Dxy(x,y) < Dxz(y,z)     % xy is closest pair
                    Cxy(x,y) = Cxy(x,y) + 1/Uxz(x,z);   
                    Cxy(y,x) = Cxy(y,x) + 1/Uxz(y,z);   %Uxz = Uyz 
                elseif Dxz(x,z) < Dxy(x,y) && Dxz(x,z) < Dxz(y,z) % xz is closest pair
                    Cxz(x,z) = Cxz(x,z) + 1/Uxy(x,y);
                    Czx(z,x) = Czx(z,x) + 1/Uxz(y,z);   %Uxz = Uyz
                else                                      % yz is closest pair
                    Cxz(y,z) = Cxz(y,z) + 1/Uxy(x,y);   %Cxz = Cyz
                    Czx(z,y) = Czx(z,y) + 1/Uxz(x,z);   %Czx = Czy
                end

            end
         end
     end
         
end

function [Cxz,Cyz,Czx] = update_coh2(Uxz,Uyz,Dxz,Dyz,Cxz,Cyz,Czx)   
% update Cohension matrix block when y == z 
[m,n] = size(Dxz);
    
     for x = 1:m
        for y = 1:n
            for z = (y+1):n
                
                if Dxz(x,y) < Dxz(x,z) && Dxz(x,y) < Dyz(y,z)     % xy is closest pair
                    Cxz(x,y) = Cxz(x,y) + 1/Uxz(x,z);   % Cxz = Cxy
                    Czx(y,x) = Czx(y,x) + 1/Uyz(y,z);   % Czx = Cyx
                elseif Dxz(x,z) < Dxz(x,y) && Dxz(x,z) < Dyz(y,z) % xz is closest pair
                    Cxz(x,z) = Cxz(x,z) + 1/Uxz(x,y);   % Uxz = Uxy
                    Czx(z,x) = Czx(z,x) + 1/Uyz(y,z);   % 
                else                                      % yz is closest pair
                    Cyz(y,z) = Cyz(y,z)+ 1/Uxz(x,y); % Uxz = Uxy
                    Cyz(z,y) = Cyz(z,y)+ 1/Uxz(x,z); % 
        
                end

            end
         end
     end
     
end


function [Cxy,Cxz,Cyz,Cyx,Czx,Czy] = update_coh3(xb, yb, zb, Uxy, Uxz, Uyz, Dxy, Dxz, Dyz)
% update Cohension matrix block when y ~= z && x ~= y
    [m,n] = size(Dxy);
    [~,p] = size(Dxz);
    Cxy = zeros(m,n);
    Cxz = zeros(m, p);
    Cyz = zeros(n, p);
    Cyx = zeros(n, m);
    Czx = zeros(p, m);
    Czy = zeros(p, n);
    
    xend = m;
    ystart = 1;
    zstart = 1;
    if (xb == yb) && (yb == zb)
        xend = m - 1;
    end
     for x = 1:xend
        if xb == yb
            ystart = x + 1;
        end
        for y = ystart:n
            if yb == zb
                zstart = y + 1;
            end
            for z = zstart:p
                if Dxy(x,y) < Dxz(x,z) && Dxy(x,y) < Dyz(y,z)     % xy is closest pair
                    Cxy(x,y) = Cxy(x,y) + 1/Uxz(x,z);
                    Cyx(y,x) = Cyx(y,x) + 1/Uyz(y,z);
                elseif Dxz(x,z) < Dxy(x,y) && Dxz(x,z) < Dyz(y,z) % xz is closest pair
                    Cxz(x,z) = Cxz(x,z) + 1/Uxy(x,y);
                    Czx(z,x) = Czx(z,x) + 1/Uyz(y,z);
                else                                      % yz is closest pair
                    Cyz(y,z) = Cyz(y,z) + 1/Uxy(x,y);
                    Czy(z,y) = Czy(z,y) + 1/Uxz(x,z);           
                end
            end
        end
     end
end

