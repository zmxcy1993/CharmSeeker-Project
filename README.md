# CharmSeeker
This project is to tune configurations (memory and workload) for serverless video processing pipelines. 
It is implemented based on the [Spearmint](https://github.com/JasperSnoek/spearmint) project.

## Prerequisites

To run the code, you first need to activate the python virtual environment in directory `python3-venv` or install the required packages through `pip3 -r requirements.txt`.

## Run SBO

Take the 2-stage pipeline for example.

``` 
cd charmseeker/examples/outer_bo_2
python ../../spearmint/sbo_main.py --grid-seed=xxx --pipeline-budget=xxx
```

You can specify the input arguments on the basis of your own needs. The configuration files for each stage can be found in `charmseeker/examples/inner_bo_sxxx`. 
You can also find the implementation codes of evaluation jobs. You may would like to change them for _specialized evaluations_.
