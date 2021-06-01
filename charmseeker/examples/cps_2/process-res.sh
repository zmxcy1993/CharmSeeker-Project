#! /bin/bash

rm -f tmp*
cat proposal.txt | awk '{print $3$4$5$6}' | grep -Eo '[0-9]+,\s*[0-9]+,\s*[0-9]+,\s*[0-9]+' | awk -F ',' '{print $1,$2,$3,$4}' > tmp

awk 'NR==FNR{var=$1" "$2;a[var]=$3" "$4} NR!=FNR{var=$1" "$2;print a[var]}' ./averages-1 tmp | sort | uniq > tmp-res-1
awk 'NR==FNR{var=$1" "$2;a[var]=$3" "$4} NR!=FNR{var=$3" "$4;print a[var]}' ./averages-2 tmp | sort | uniq > tmp-res-2

search_cost=0
for i in 1 2
do
	tmp_res=`awk 'BEGIN{sum=0}{sum+=$1}END{print sum}' tmp-res-$i`
	search_cost=`awk -v var1="$search_cost" -v var2="$tmp_res" 'BEGIN{print var1 + var2}'`
done

echo $search_cost

counts2=`wc -l tmp-res-2|awk '{print $1}'`

budget=$1

cat best_job_and_result.txt |awk 'NR==1 {print $3}' >> one-layer-part-$budget

cat best_job_and_result.txt | awk 'NR==4 {print}' | grep -Eo '[0-9]+,\s*[0-9]+,\s*[0-9]+,\s*[0-9]+' | awk -F ',' '{print $1,$2,$3,$4}' >> configs-$budget
awk -v var="$counts2" 'NR==FNR {a[NR]=$0} NR!=FNR {for(i=1;i<=var;i++) print $0,a[i]}' tmp-res-2 tmp-res-1 > tmp-12
awk '{print $0,$1+$3,$2+$4}' tmp-12 > tmp-pipeline-12
awk -v var="$budget" '$5<=var{print}' tmp-pipeline-12 | sort -k 6,6g | head -1 | awk '{print $6/1000}' >> one-layer-all-$budget

rm tmp*

