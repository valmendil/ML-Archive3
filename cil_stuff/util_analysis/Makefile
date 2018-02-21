# Makefile : creates 'analysis' images from predictions and groundtruth
#
# recommended usage: make -j8

NUMS := \
    01 02 03 04 05 06 07 08 09 10 \
    11 12 13 14 15 16 17 18 19 20 \
    21 22 23 24 25 26 27 28 29 30 \
    31 32 33 34 35 36 37 38 39 40 \
    41 42 43 44 45 46 47 48 49 50

all :

clean :
	rm -rf analysis

analysis :
	mkdir $@

define analyze

all : analysis/test_$(1).png

analysis/test_$(1).png : \
    predictions/prediction_0$(1).png \
    util/analyze \
    | analysis
	util/analyze \
	    data/extra_groundtruth/test_$(1).png \
	    predictions/prediction_0$(1).png \
	    $$@

endef

$(foreach x,$(NUMS),$(eval $(call analyze,$(x))))
