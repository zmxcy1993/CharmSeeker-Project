#! /bin/bash

for budget in 0.542808 0.543428 0.547254 0.551489 0.556283 0.577744
do
    for iteration in {1..20}
    do
    	../clean.sh

    	python ../../spearmint/sbo_main.py --grid-seed=1 --pipeline-budget=$budget

    	for stage in 1 2 3 4 5
    	do
    		cat ../inner_bo_s"$stage"_p5/proposal.txt | awk '{print $4$5}' | grep -o -E '[0-9]+,\s[0-9]+' | awk -F ',' '{print $1,$2}' > tmp-$i
    		awk 'NR==FNR{var=$1" "$2;a[var]=$3" "$4} NR!=FNR{var=$1" "$2;print a[var]}' ../inner_bo_s"$stage"_p5/averages tmp-$i > tmp-res-$i
    		awk 'BEGIN{sum=0}{sum+=$1}END{printf sum" "}' tmp-res-$i >> search-cost-$budget
    	done
    	awk 'BEGIN{print}' >> search-cost-$budget
    	
    	awk 'NR==FNR {a[NR]=$0} NR!=FNR {for(i=1;i<=12;i++) print $0,a[i]}' tmp-res-5 tmp-res-4 > tmp-45
    	awk 'NR==FNR {a[NR]=$0} NR!=FNR {for(i=1;i<=12*12;i++) print $0,a[i]}' tmp-45 tmp-res-3 > tmp-345
		awk 'NR==FNR {a[NR]=$0} NR!=FNR {for(i=1;i<=12*12*12;i++) print $0,a[i]}' tmp-345 tmp-res-2 > tmp-2345
		awk 'NR==FNR {a[NR]=$0} NR!=FNR {for(i=1;i<=12*12*12*12;i++) print $0,a[i]}' tmp-2345 tmp-res-1 > tmp-12345
	
		awk '{print $0,$1+$3+$5+$7+$9,$2+$4+$6+$8+$10}' tmp-12345 > tmp-pipeline-12345
		awk -v var="$budget" '$11<=var{print}' tmp-pipeline-12345| sort -k 12,12g | head -1 | awk '{print $12/1000}' >> sbo-$budget
    	rm tmp*
    done
done






