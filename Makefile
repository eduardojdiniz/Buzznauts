# Targets that do not create files
.PHONY: all clean_baseline flake8 test

VDIR?='data/stimuli/videos'
SDIR?='models/baseline'
TRACK?='all'
FMRI_DIR?='data/fmri'
LAYER?='layer_5'
RESULT_DIR?='results/alexnet' + "/" + "$(LAYER)"

# Rule to the entire pipeline
all:


flake8:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 flake8 --ignore N802,N806 `find . -name \*.py | grep -v setup.py | grep -v /doc/`; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

test:
	py.test --pyargs Buzznauts --cov-report term-missing --cov=Buzznauts

# Rule to remove all generated output by baseline model
clean_baseline:
	rm -rf models/baseline

baseline:
	python analysis/baseline/s01_generate_features_alexnet.py -vdir $(VDIR) -sdir $(SDIR)
ifeq ($(TRACK), "all")
	python analysis/baseline/s03_generate_all_results.py -fd $(FMRI_DIR) -t mini_track -l $(LAYER)
	python Buzznauts/app/prepare_submission.py -rd $(RESULT_DIR) -t mini_track
	python analysis/baseline/s03_generate_all_results.py -fd $(FMRI_DIR) -t full_track -l $(LAYER)
	python Buzznauts/app/prepare_submission.py -rd $(RESULT_DIR) -t full_track
else
	python analysis/baseline/s03_generate_all_results.py -fd $(FMRI_DIR) -t $(TRACK) -l $(LAYER)
	python Buzznauts/app/prepare_submission.py -rd $(RESULT_DIR) -t $(TRACK)
endif
