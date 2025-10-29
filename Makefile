export OMP_NUM_THREADS := 1
export NANOCHAT_BASE_DIR ?= $(HOME)/.cache/nanochat
EVAL_BUNDLE_URL := https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip

nanochat-end-to-end: report-reset prepare-dataset train-eval-tokenizer prepare-evalsets pretrain-d20

report-reset:
# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
	python -m nanochat.report reset

prepare-dataset:
# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
	python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
	python -m nanochat.dataset -n 240

train-eval-tokenizer:
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
	python -m scripts.tok_train --max_chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
	python -m scripts.tok_eval

.ONESHELL:
prepare-evalsets:
# Download the eval_bundle from s3 to evaluate CORE metric during training (~162MB)
	if [ ! -d $(NANOCHAT_BASE_DIR)/eval_bundle ]; then
		curl -L -o eval_bundle.zip $(EVAL_BUNDLE_URL)
		unzip -q eval_bundle.zip
		rm eval_bundle.zip
		mv eval_bundle $(NANOCHAT_BASE_DIR)
	fi

.ONESHELL:
base-train:
	GPU_MODEL=`nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | awk '{print $$2}'`
	DATE=`date +"%y%m%d"`
	NLAYER=20
	NGPU=$$(nvidia-smi --list-gpus | wc -l)
	PREC=bf16
	RUN_ID=$${DATE}__$${PREC}_d$${NLAYER}_base_train_$${GPU_MODEL}x$${NGPU}
	if [ -n "$$WANDB_API_KEY" ]; then
		RUN_ARG="--run=$${RUN_ID}"
	else
		RUN_ARG=""
	fi
	@echo "Start pretrain $${RUN_ID}"
	sleep 3
	python -m torch.distributed.run --standalone --nproc_per_node=$${NGPU} \
		-m scripts.base_train -- \
		--depth=$${NLAYER} \
		--prec=$${PREC} \
		$${RUN_ARG}

base-loss:
# evaluate the model on a larger chunk of train/val data and draw some samples
	python -m torch.distributed.run --standalone \
		--nproc_per_node=$$(nvidia-smi --list-gpus | wc -l) \
		-m scripts.base_loss

base-eval:
# evaluate the model on CORE tasks
	python -m torch.distributed.run --standalone \
		--nproc_per_node=$$(nvidia-smi --list-gpus | wc -l) \
		-m scripts.base_eval