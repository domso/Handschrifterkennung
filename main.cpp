#include <iostream>
#include "src/sample_set.h"

int main(int argc, char** argv) {
    auto v = data::sample_set::load<uint8_t>("../train-images.idx3-ubyte", "../train-labels.idx1-ubyte");

    for (data::sample<uint8_t>& s : v) {

    }

    v[21835].store("out");
    //v[0].print();
    //v[0].store("test.pgm");

    return 0;
}
