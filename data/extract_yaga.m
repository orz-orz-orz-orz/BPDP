%lines = readlines("APLAWDW_egg.txt");
lines = readlines("CMU_ARCTICS_SJB30.txt");
for i = 1:length(lines)
    line = lines(i);
    path = split(line, '/');
    % out_file = append("APLAWDW_EGG_YAGAV_GCI/", path(5), ".txt");
    out_file = append("CMU_ARCTICS_SJB30_YAGA/", path(2), "/", path(3), ".txt");
    [y, fs, wmode, fidx] = v_readwav(line);
    y = y(:, 1);
    gci = yaga(y, fs);
    writematrix(gci', out_file);
end