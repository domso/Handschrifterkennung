#pragma once

#define SUDOKU_FIELD_WIDTH 9
#define SUDOKU_FIELD_HEIGHT 9
#define SUDOKU_FIELD_SIZE 81
#include <string>

class Sudoku {
public:

    //creates an empty Sudoku
    Sudoku();

    // creates an SUdoku with the textFormat containing the field
    explicit Sudoku(const std::string& textFormat);

    // prints the current Sudoku
    void print() const;

    // checks, whether the sudoku is valid (not working)
    bool is_valid() const;

    // checks, whether the sudoku is complete
    bool is_complete() const;

    // accessses the Sudoku at position pos
    char operator[](const int pos) const;

    bool operator==(const Sudoku& s) const;

    // sets value on position pos (row-major)
    void set(const int pos, const int value);

    // returns next free position
    int getNextPos() const;

    // tries to insert value on pos
    bool tryInsert(const int pos, const int value);

    // checks if value can be used on pos
    bool checkPos(const int pos, const int value) const;
private:
    char field_[SUDOKU_FIELD_SIZE];


};

