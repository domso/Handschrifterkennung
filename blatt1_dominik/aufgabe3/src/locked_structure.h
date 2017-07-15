#ifndef locked_structure_h
#define locked_structure_h

#include <mutex>
#include <condition_variable>

template <typename T>
class locked_structure {
public:
    locked_structure() : counter_(0) {

    }

    T obj;

    void lock() {
        return mutex_.lock();
    }

    void unlock() {
        return mutex_.unlock();
    }

    bool try_lock() {
        return mutex_.try_lock();
    }
private:
    int counter_;
    std::mutex mutex_;
};

#endif
