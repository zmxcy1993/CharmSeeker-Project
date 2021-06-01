#! /bin/bash

for budget in 0.376867 0.377238 0.379628 0.382663 0.385794 0.399147
do
	for iteration in {1..20}
	do
		../clean.sh

		python ../../spearmint/sbo_main.py --grid-seed=1 --pipeline-budget=$budget

		for stage in 1 2 3
		do
			cat ../inner_bo_s"$stage"/proposal.txt | awk '{print $4$5}' |  grep -o -E '[0-9]+,\s[0-9]+' | awk -F ',' '{print $1,$2}' > tmp-$i
			awk 'NR==FNR{var=$1" "$2;a[var]=$3" "$4} NR!=FNR{var=$1" "$2;print a[var]}' ../inner_bo_s"$stage"/averages tmp-$i > tmp-res-$i
			awk 'BEGIN{sum=0}{sum+=$1}END{printf sum" "}' tmp-res-$i >> search-cost-$budget
		done

		awk 'BEGIN{print}' >> search-cost-$budget	
		awk 'NR==FNR {a[NR]=$0} NR!=FNR {for(i=1;i<=20;i++) print $0,a[i]}' tmp-res-3 tmp-res-2 > tmp-23
		awk 'NR==FNR {a[NR]=$0} NR!=FNR {for(i=1;i<=400;i++) print $0,a[i]}' tmp-23 tmp-res-1 > tmp-123
		awk '{print $0,$1+$3+$5,$2+$4+$6}' tmp-123 > tmp-pipeline-123
		awk -v var="$budget" '$7<=var{print}' tmp-pipeline-123 | sort -k 8,8g | head -1 | awk '{print $8/1000}' >> sbo-$budget

		rm tmp*
	done
done



	



	
