for input in *.m4a; do
	output="${input%.m4a}_test.m4a"
	ffmpeg -ss 10 -t 45 -i "$input" "$output"
done
