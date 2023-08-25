for j in 1 2 3 4 5;
do
	python train_bus_dk_vat.py --num-labeled 878 \
	--total-steps 5500 --eval-step 110 --warmup 220 --dkdecline linear \
	--batch-size 8 --lr 0.0046875 --mu 30 --lr-drop-iter 1540 3190 4840 \
	--threshold 0.9 --dkthreshold 0.95 \
	--labeledpath ../data_split/28/labeled_images_20_9.pth \
	--unlabeledpath ../data_split/28/unlabeled_images_80_9.pth \
	--out bus@878_8mu30_${j}_dk_vat
	$@
done