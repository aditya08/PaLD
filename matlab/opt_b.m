function b_opt = opt_b(func,D,range,len)
% manually determine the size of b in block related algorithms for PaLD
% func is the block algorithm function, 
% D is the distance matrix
% range is the search range of b
% len is the search step length for b
opt_btime = Inf;

for i = 1:range
   b = i*len+300;
   tic
   [~,~] = func(D,1,b);
   b_time = toc ;
   if b_time < opt_btime
       b_opt = b;
       opt_btime = b_time;
   end
    
end

end