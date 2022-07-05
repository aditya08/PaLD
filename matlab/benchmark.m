function benchmark(edge_len,cache_size)
fileID = fopen('dist_mat.bin');
mat_1d = fread(fileID,edge_len*edge_len,'double');
fclose(fileID);
D = reshape(mat_1d,[edge_len,edge_len]);
maxNumCompThreads(1);
t1 = timeit(@()pald_orig(D,1));
fprintf("Orig time: %.2f\n",t1);
t2 = timeit(@()pald_opt(D,1,cache_size));
fprintf("Opt time: %.2f\n",t2);
end
