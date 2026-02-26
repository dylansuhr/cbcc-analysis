.PHONY: all clean eda eda2

DATA = data/booking-report-23-24-25.csv
OUTPUT_DIR = output/figures

all: eda

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

eda: $(OUTPUT_DIR) $(DATA)
	python src/eda.py

eda2: $(DATA)
	python src/eda2.py

clean:
	rm -rf $(OUTPUT_DIR)/*.png output/eda_report.md
