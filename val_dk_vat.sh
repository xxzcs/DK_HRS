for j in 1 2 3 4 5;
do
	python val_bus_projection.py --resume bus@878_8mu30_${j}_dk_vat \
	--csvname test28_fixmatch_dk_vat_${j}.csv
 	$@
done