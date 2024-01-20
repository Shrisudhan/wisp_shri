max_res=2048
code_bit=19
objects=("sphere") # "lego" "ship" "drums" "chair" "hotdog" "material" "mic" "ficus"
export CUDA_VISIBLE_DEVICES=1

for object in "${objects[@]}"
do
	echo "Training INGP with ${max_res} as maximum resolution with codebook bit resolution ${code_bit} for i-NGP"
	python app/nerf/main_nerf.py \
		--config "app/nerf/configs/nerf_hash.yaml" \
		--interactive False \
		--dataset.dataset-path "/data/nerf_dataset/blender/${object}/" \
		--trainer.max-epochs 500 \
		--trainer.render-every 20 \
		--trainer.valid-every 500 \
		--trainer.rgb-loss-type 'l1' \
		--trainer.optimizer.lr 1.0e-3 \
		--trainer.grid-lr-weight 500 \
		--grid.codebook-bitwidth ${code_bit} \
		--grid.max-grid-res ${max_res} \
		--trainer.exp-name "InstantNGP_uniform_sample/${object}"
done