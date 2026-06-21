# Wiki-Link Retrieval — SIGIR 2027 Paper Build
#
# Uses tectonic (zero-config LaTeX engine) with local cache.
# Tectonic binary: .tools/tectonic/tectonic (symlinked from vamos)

TECTONIC     ?= .tools/tectonic/tectonic
PAPER_DIR    := paper
PAPER_TEX    := main.tex
PAPER_PDF    := main.pdf
CACHE_DIR    := $(abspath $(PAPER_DIR)/.tectonic-cache)

.PHONY: paper paper-clean paper-watch check-tectonic help

## Build the paper PDF
paper: check-tectonic
	@mkdir -p $(CACHE_DIR)
	cd $(PAPER_DIR) && XDG_CACHE_HOME=$(CACHE_DIR) \
		$(abspath $(TECTONIC)) $(PAPER_TEX)
	@echo "==> $(PAPER_DIR)/$(PAPER_PDF) built successfully"

## Build with --watch (auto-rebuild on save)
paper-watch: check-tectonic
	@mkdir -p $(CACHE_DIR)
	cd $(PAPER_DIR) && XDG_CACHE_HOME=$(CACHE_DIR) \
		$(abspath $(TECTONIC)) --watch $(PAPER_TEX)

## Remove build artifacts
paper-clean:
	rm -f $(PAPER_DIR)/$(PAPER_PDF)
	rm -rf $(CACHE_DIR)

## Verify tectonic is available
check-tectonic:
	@test -x $(TECTONIC) || { echo "tectonic not found at $(TECTONIC)"; exit 1; }

## Show help
help:
	@echo "make paper         Build paper PDF"
	@echo "make paper-watch   Auto-rebuild on file save"
	@echo "make paper-clean   Remove PDF and cache"
	@echo "make check-tectonic Verify tectonic binary"
