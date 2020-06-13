#include <stdint.h>

void resizeThread(uint8_t* src, uint8_t* dst, int newSize, int oldSize, float scale, int startI, int endI);
uint8_t* resize(uint8_t* src, int newSize, int oldSize, float scale);
