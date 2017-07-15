#include <vector>
#include "thread_pool.h"

#include "locked_structure.h"

#include "sudoku.h"

class Solver {
public:

    Solver(Sudoku& s, locked_structure<std::vector<Sudoku>>& results) : s_(s), results_(results) {

    }

    void operator()(Thread_Pool<Solver>* threadPool) {
        if (s_.is_complete()) {
            std::unique_lock<locked_structure<std::vector<Sudoku>>> ul(results_);
            results_.obj.push_back(s_);
            return;
        }

        int pos = s_.getNextPos();

        if (pos < 0) {
            return;
        }

        // pos required to be >= 0

        for (int number = 1; number < 10; number++) {
            s_.set(pos, 0);

            if (s_.tryInsert(pos, number)) {

                // create new tasks for every new combination
                Solver newSolver(s_, results_);
                threadPool->insert(newSolver);
            }
        }

        return;
    }

private:
    Sudoku s_;
    locked_structure<std::vector<Sudoku>>& results_;
};

/* old and ugly version

#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "thread_pool.h"

#include "sudoku.h"


class Solver {
public:

    Solver(const Sudoku s, const int numThreads, std::vector<Sudoku>& results) : numThreads_(numThreads), currentRunning_(numThreads), stack_(numThreads), mutex_(numThreads), results_(results) {
        stack_[0].push_back(s);
    }

    bool task(Thread_Pool<Solver>* threadPool, const int id) {
        std::unique_lock<std::mutex> ul(mutex_[id]);

        if (stack_[id].empty()) {
            return true;
        }

        Sudoku s = stack_[id][stack_[id].size() - 1];
        stack_[id].resize(stack_[id].size() - 1);

        ul.unlock();

        if (s.is_complete()) {
            ResultMutex_.lock();
            results_.push_back(s);
            ResultMutex_.unlock();
            return false;
        }

        int pos = s.getNextPos();
        if (pos < 0) {
            return false;
        }
        // pos required to be >= 0

        for (int number = 1; number < 10; number++) {
            s.set(pos, 0);

            if (s.tryInsert(pos, number)) {
                int set = (id + 1) % numThreads_;
                mutex_[set].lock();
                stack_[set].push_back(s);
                mutex_[set].unlock();
            }
        }
        return false;

    }

    bool wait() {
        std::unique_lock<std::mutex> ul(waitMutex_);

        while (currentRunning_.load() != 0) {
            cond_.wait(ul);
        }

        return !results_.empty();
    }

    void operator()(Thread_Pool<Solver>* threadPool) {
        bool result = true;
        do {
            result = true;
            for (int i = 0; i < numThreads_; i++) {
                result &= task(threadPool, i);
            }
        } while (!result);

        currentRunning_.fetch_sub(1);
        cond_.notify_all();
    }
private:

    int numThreads_;
    std::atomic<int> currentRunning_;

    std::mutex waitMutex_;

    std::vector<std::vector<Sudoku>> stack_;
    std::vector<std::mutex> mutex_;

    std::vector<Sudoku>& results_;
    std::mutex ResultMutex_;
    std::condition_variable cond_;
};

*/

