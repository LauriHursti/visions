#include "constants.h"

bool oppositesTest(uint8_t center, uint8_t outers[], int threshold);
bool connectivityTest(uint8_t center, uint8_t outers[], uint8_t inners[], int threshold);
int* getKpStats(uint8_t center, uint8_t outers[], int threshold);
void parseKps(uint8_t* im, int* icollector, int size, float scale, int threshold, int istart, int iend);
void multiScaleKps(uint8_t* image, int* icollector, int size, int scales, int threshold);
