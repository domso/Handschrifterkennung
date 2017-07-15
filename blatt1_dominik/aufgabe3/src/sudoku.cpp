#include <iostream>
#include "sudoku.h"


Sudoku::Sudoku() {
    for (int i = 0; i < SUDOKU_FIELD_SIZE; i++) {
        field_[i] = 0;
    }
}

Sudoku::Sudoku(const std::string& textFormat) {
    int length = textFormat.length();

    if (length >= SUDOKU_FIELD_SIZE) {
        length = SUDOKU_FIELD_SIZE;
    }

    int i;
    int tmp = 0;

    for (i = 0; i < length; i++) {
        if (textFormat[i] == ' ') {
            field_[i] = 0;
            tmp++;
        } else {
            field_[i] = textFormat[i] - 48;
        }
    }

    for (; i < SUDOKU_FIELD_SIZE; i++) {
        field_[i] = 0;
    }
}


void Sudoku::print() const {
    int n = 0;
    std::cout << "___________________" << std::endl;

    for (int i = 0; i < SUDOKU_FIELD_SIZE; i++) {
        if (field_[i] != 0) {
            std::cout << "|" << (int) field_[i];
        } else {
            std::cout << "| ";
        }

        n++;

        if (n == SUDOKU_FIELD_WIDTH) {
            n = 0;
            std::cout << "|" << std::endl;
        }
    }
}

bool Sudoku::is_valid() const {
    for (int i = 0; i < SUDOKU_FIELD_SIZE; i++) {
        if (!checkPos(i, field_[i]) || field_[i] == 0) {
            return false;
        }
    }

    return true;
}

bool Sudoku::is_complete() const {
    for (int i = 0; i < SUDOKU_FIELD_SIZE; i++) {
        if (field_[i] == 0) {
            return false;
        }
    }

    return true;
}

char Sudoku::operator[](const int pos) const {
    return field_[pos];
}

bool Sudoku::operator== (const Sudoku& s) const {
    for (int i = 0; i < SUDOKU_FIELD_SIZE; i++) {
        if (field_[i] != s.field_[i]) {
            return false;
        }
    }

    return true;
}

void Sudoku::set(const int pos, const int value) {
    field_[pos] = value;
}

int Sudoku::getNextPos() const {
    for (int i = 0; i < SUDOKU_FIELD_SIZE; i++) {
        if (field_[i] == 0) {
            return i;
        }
    }

    return -1;
}

bool Sudoku::tryInsert(const int pos, const int value) {
    if (checkPos(pos, value)) {
        field_[pos] = value;
        return true;
    }

    return false;
}

bool Sudoku::checkPos(const int pos, const int value) const {
    int x = pos % SUDOKU_FIELD_WIDTH;
    int y = pos / SUDOKU_FIELD_WIDTH;

    for (int i = 0; i < SUDOKU_FIELD_SIZE; i++) {
        int localX = i % SUDOKU_FIELD_WIDTH;
        int localY = i / SUDOKU_FIELD_WIDTH;

        int localBlockX = localX / 3;
        int localBlockY = localY / 3;

        int blockX = x / 3;
        int blockY = y / 3;

        if (localX == x || localY == y || (localBlockX == blockX && localBlockY == blockY)) {
            if (value == field_[i] && i != pos) {
                return false;
            }
        }
    }

    return true;
}
