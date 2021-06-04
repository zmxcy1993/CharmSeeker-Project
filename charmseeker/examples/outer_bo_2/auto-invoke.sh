#! /bin/bash


for budget in 0.361308 0.361601 0.363456 0.36592 0.368373 0.38177
do
	for iteration in {1..20}
	do
		  ../clean.sh

      python ../../spearmint/sbo_main.py --grid-seed=1 --pipeline-budget=$budget

    	for stage in 1 2
    	do
            # get the proposed sub-configurations for each stage
    		cat ../inner_bo_s"$stage"/proposal.txt | awk '{print $4$5}' | grep -o -E '[0-9]+,\s[0-9]+' | awk -F ',' '{print $1,$2}' > tmp-$stage
            # get the corresponding durations and costs for the sub-configurations of each stage
    		awk 'NR==FNR {var=$1" "$2; a[var]=$3" "$4} NR!=FNR{var=$1" "$2; print a[var]}' ../inner_bo_s"$stage"/averages tmp-$stage > tmp-res-$stage
            # get the seach cost for the sub-configurations of each stage
    		awk 'BEGIN{sum=0}{sum += $1}END{printf sum" "}' tmp-res-$stage >> search-cost-$budget
    	done

        # back up the pipeline seach cost for the current iteration
    	awk 'BEGIN{print}' >> search-cost-$budget

        # get cartesian products of sub-configurations
        awk 'NR==FNR {a[NR]=$0} NR!=FNR {for(i=1;i<=20;i++) print $0,a[i]}' tmp-res-2 tmp-res-1 > tmp-12
        awk '{print $0,$1+$3,$2+$4}' tmp-12 > tmp-pipeline-12
        # get the searched optimal configuration, i.e., the configuration with the minimum duration give the budget.
        awk -v var="$budget" '$5 <= var{print}' tmp-pipeline-12 | sort -k 6,6g | head -1 | awk '{$6/1000}' >> sbo-$budget
        
        rm tmp*
	done
done


