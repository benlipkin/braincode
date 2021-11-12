mkdir -p ../plots ../tables/raw ../tables/latex ../stats/raw ../stats/latex
python scores.py
python stats.py
python plots.py
python tables.py
cd supplemental/property_corrs
python code_property_corrs.py
cd ../model_dims
python model_dim_expt.py
cd ../../
